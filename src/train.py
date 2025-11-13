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
import sys
import yaml


# ensure repository root is on sys.path based on this file's location (works regardless of cwd)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)  # add repo root so `src` package can be imported
from src.process_datasets import get_web_feats
from src.process_datasets import aggregate_features, create_data
from src.eval import read_baseline_auc


# Load the YAML config file
with open(os.path.join(repo_root, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)


def run():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    X_train, y_train, X_test, y_test, train, test = create_data(repo_root, config)
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
    scale_pos_weight = round(y_train.value_counts()[0] / y_train.value_counts()[1])
    base = XGBClassifier(
        eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight
    )
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
    preds = (proba >= config["threshold"]).astype(int)

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

    print(f"\nClassification report (threshold={config['threshold']}):")
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
