import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier
import joblib
from scipy import stats
import json
from datetime import datetime
from process_datasets import get_web_feats
import yaml

# Load the YAML config file
with open("wellco_churn\\config.yaml", "r") as f:
    config = yaml.safe_load(f)


def aggregate_features(data_dir):
    # Helper to accept either canonical filenames or ones prefixed with 'test_'
    def pick(path_dir, name):
        p1 = os.path.join(path_dir, name)
        p2 = os.path.join(path_dir, f"test_{name}")
        if os.path.exists(p1):
            return p1
        if os.path.exists(p2):
            return p2
        return p1  # default (may not exist)

    app_path = pick(data_dir, "app_usage.csv")
    web_path = pick(data_dir, "web_visits.csv")
    claims_path = pick(data_dir, "claims.csv")
    labels_path = pick(data_dir, "churn_labels.csv")

    # Read files (some files may be missing in tiny test fixtures; handle gracefully)
    if os.path.exists(app_path):
        app = pd.read_csv(app_path)
        app_feats = app.groupby("member_id").size().rename("session_count")
    else:
        app_feats = pd.Series(dtype=int, name="session_count")

    if os.path.exists(web_path):
        web = pd.read_csv(web_path)

        try:

            web_feats = get_web_feats(web_path, config)
            # normalize return to Series indexed by member_id with a sensible name
            if isinstance(web_feats, pd.DataFrame):
                if "member_id" in web_feats.columns:
                    web_feats = web_feats.set_index("member_id")
                # if single-column DF, convert to Series
                if web_feats.shape[1] == 1:
                    web_feats = web_feats.iloc[:, 0]
            if isinstance(web_feats, pd.Series):
                if web_feats.name is None:
                    web_feats = web_feats.rename("web_visit_count")
        except Exception:
            # fallback to simple aggregation if get_web_feats is not available/applicable
            web = pd.read_csv(web_path)
            web_feats = web.groupby("member_id").size().rename("web_visit_count")
    else:
        web_feats = pd.Series(dtype=int, name="web_visit_count")

    if os.path.exists(claims_path):
        claims = pd.read_csv(claims_path)
        claims_count = claims.groupby("member_id").size().rename("claims_count")
        # flags for some ICD codes of interest
        codes = ["E11.9", "I10", "Z71.3"]
        for code in codes:
            col = f"has_{code.replace('.', '_')}"
            flag = claims["icd_code"].fillna("").str.startswith(code)
            flag_series = claims.loc[flag].groupby("member_id").size().rename(col)
            # convert counts to 1
            flag_series = (flag_series >= 1).astype(int)
            claims_count = (
                claims_count.to_frame().join(flag_series, how="left")
                if isinstance(claims_count, pd.Series)
                else claims_count.join(flag_series, how="left")
            )
        claims_feats = claims_count
    else:
        # empty frame with default columns
        claims_feats = pd.DataFrame(
            columns=["claims_count", "has_E11_9", "has_I10", "has_Z71_3"]
        )

    # Combine feature frames
    feats = pd.concat([app_feats, web_feats, claims_feats], axis=1)
    feats = feats.fillna(0)

    # Read labels
    labels = pd.read_csv(labels_path)
    labels = labels.set_index("member_id")

    # Merge
    df = labels.join(feats, how="left")
    df = df.fillna(0)

    return df.reset_index()


def read_baseline_auc(baseline_path):
    if not os.path.exists(baseline_path):
        return None
    with open(baseline_path, "r") as f:
        for line in f:
            if "=" in line:
                parts = line.strip().split("=")
                try:
                    return float(parts[1])
                except:
                    continue
    return None


def run():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_dir = os.path.join(repo_root, "data", "train")
    test_dir = os.path.join(repo_root, "data", "test")

    print("Aggregating train features...")
    train = aggregate_features(train_dir)
    print("Aggregating test features...")
    test = aggregate_features(test_dir)

    feature_cols = [
        c
        for c in train.columns
        if c not in ["member_id", "signup_date", "churn", "outreach"]
    ]

    X_train = train[feature_cols]
    y_train = train["churn"].astype(int)
    X_test = test[feature_cols]
    y_test = test["churn"].astype(int)

    print(f"Training data: {len(X_train)} samples; test data: {len(X_test)} samples")

    # Determine sensible number of CV folds based on class balance
    desired_splits = 5
    min_class_count = int(y_train.value_counts().min())
    n_splits = min(desired_splits, max(2, min_class_count))
    print(f"Using StratifiedKFold with n_splits={n_splits}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Hyperparameter search space (small and fast)
    param_distributions = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    base = XGBClassifier(eval_metric="logloss", random_state=42)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="roc_auc",
        cv=skf,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )

    print("Running hyperparameter search (RandomizedSearchCV)...")
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    best_model = search.best_estimator_

    # Evaluate CV scores (per-fold) for the best estimator to compute mean and 95% CI
    print("Computing cross-validated AUCs for best estimator...")
    cv_scores = cross_val_score(
        best_model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
    )
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores, ddof=1)) if len(cv_scores) > 1 else 0.0
    cv_n = len(cv_scores)
    # 95% t-based CI
    if cv_n > 1:
        t_value = float(stats.t.ppf(1 - 0.025, df=cv_n - 1))
        ci_half = t_value * (cv_std / np.sqrt(cv_n))
    else:
        ci_half = 0.0

    print(f"CV ROC-AUC: mean={cv_mean:.4f}, std={cv_std:.4f}, n={cv_n}")
    print(f"95% CI: [{cv_mean - ci_half:.4f}, {cv_mean + ci_half:.4f}]")

    # Refit best model on full training set (RandomizedSearchCV already refits by default)
    final_model = best_model

    # Save model artifact
    models_dir = os.path.join(repo_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(
        models_dir, f"best_model_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.joblib"
    )
    joblib.dump(final_model, model_path)
    print(f"Saved best model to: {model_path}")

    # Evaluate on test set
    proba = final_model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    try:
        test_auc = roc_auc_score(y_test, proba)
    except Exception as e:
        test_auc = None
        print("Could not compute test AUC:", e)

    print("\nModel results on test set:")
    if test_auc is not None:
        print(f"  ROC-AUC (xgboost) = {test_auc:.4f}")
    else:
        print("  ROC-AUC not available")

    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_test, preds, zero_division=0))

    # Save test predictions
    outputs_dir = os.path.join(repo_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    preds_df = pd.DataFrame(
        {"member_id": test["member_id"], "proba": proba, "pred": preds}
    )
    preds_path = os.path.join(outputs_dir, "test_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved test predictions to: {preds_path}")

    baseline_path = os.path.join(repo_root, "data", "auc_baseline_test.txt")
    baseline_auc = read_baseline_auc(baseline_path)
    if baseline_auc is not None and test_auc is not None:
        print(f"\nBaseline AUC = {baseline_auc:.4f}")
        diff = test_auc - baseline_auc
        print(f"  Difference (model - baseline) = {diff:.4f}")
    else:
        print("\nBaseline AUC not found or test AUC not computed.")

    # Save a small report with CV stats and test AUC
    report = {
        "cv_mean_auc": cv_mean,
        "cv_std_auc": cv_std,
        "cv_n": cv_n,
        "cv_95ci_lower": cv_mean - ci_half,
        "cv_95ci_upper": cv_mean + ci_half,
        "test_auc": test_auc,
        "baseline_auc": baseline_auc,
        "model_path": model_path,
        "predictions_path": preds_path,
    }
    with open(os.path.join(outputs_dir, "run_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved run report to: {os.path.join(outputs_dir, 'run_report.json')}")


if __name__ == "__main__":
    run()
