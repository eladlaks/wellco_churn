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
from src.process_datasets import aggregate_features
from src.eval import read_baseline_auc

# Load the YAML config file
with open(os.path.join(repo_root, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)


def train_two_model_uplift(X_train, y_train, X_test, y_test, treatment):
    """
    Train two separate XGBoost models (one for treatment, one for control)
    to estimate heterogeneous treatment effect (uplift).

    Args:
        X_train: Feature matrix for training
        y_train: Target for training
        X_test: Feature matrix for testing
        y_test: Target for testing
        treatment: Binary treatment indicator for all samples

    Returns:
        dict with model performance and uplift estimates
    """

    # Split by treatment status
    treatment_mask_train = treatment.iloc[: len(X_train)].values == 1
    control_mask_train = treatment.iloc[: len(X_train)].values == 0

    treatment_mask_test = treatment.iloc[len(X_train) :].values == 1
    control_mask_test = treatment.iloc[len(X_train) :].values == 0

    X_train_treatment = X_train[treatment_mask_train]
    y_train_treatment = y_train[treatment_mask_train]
    X_train_control = X_train[control_mask_train]
    y_train_control = y_train[control_mask_train]

    X_test_treatment = X_test[treatment_mask_test]
    y_test_treatment = y_test[treatment_mask_test]
    X_test_control = X_test[control_mask_test]
    y_test_control = y_test[control_mask_test]

    print(f"\nTreatment group in train: {len(X_train_treatment)} samples")
    print(f"Control group in train: {len(X_train_control)} samples")
    print(f"Treatment group in test: {len(X_test_treatment)} samples")
    print(f"Control group in test: {len(X_test_control)} samples")

    results = {"treatment_group": {}, "control_group": {}}

    # Train treatment model
    if len(X_train_treatment) > 0:
        print("\n--- Training treatment model ---")
        scale_pos_weight = (
            (y_train_treatment == 0).sum() / (y_train_treatment == 1).sum()
            if (y_train_treatment == 1).sum() > 0
            else 1
        )
        treatment_model = XGBClassifier(
            eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight
        )
        treatment_model.fit(X_train_treatment, y_train_treatment)

        if len(X_test_treatment) > 0:
            proba_treatment = treatment_model.predict_proba(X_test_treatment)[:, 1]
            auc_treatment = roc_auc_score(y_test_treatment, proba_treatment)
            results["treatment_group"]["auc"] = float(auc_treatment)
            results["treatment_group"]["n_samples"] = len(X_test_treatment)
            print(f"Treatment model test AUC: {auc_treatment:.4f}")
        else:
            proba_treatment = None
            results["treatment_group"]["auc"] = None
            results["treatment_group"]["n_samples"] = 0
    else:
        treatment_model = None
        proba_treatment = None
        results["treatment_group"]["auc"] = None
        results["treatment_group"]["n_samples"] = 0

    # Train control model
    if len(X_train_control) > 0:
        print("\n--- Training control model ---")
        scale_pos_weight = (
            (y_train_control == 0).sum() / (y_train_control == 1).sum()
            if (y_train_control == 1).sum() > 0
            else 1
        )
        control_model = XGBClassifier(
            eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight
        )
        control_model.fit(X_train_control, y_train_control)

        if len(X_test_control) > 0:
            proba_control = control_model.predict_proba(X_test_control)[:, 1]
            auc_control = roc_auc_score(y_test_control, proba_control)
            results["control_group"]["auc"] = float(auc_control)
            results["control_group"]["n_samples"] = len(X_test_control)
            print(f"Control model test AUC: {auc_control:.4f}")
        else:
            proba_control = None
            results["control_group"]["auc"] = None
            results["control_group"]["n_samples"] = 0
    else:
        control_model = None
        proba_control = None
        results["control_group"]["auc"] = None
        results["control_group"]["n_samples"] = 0

    # Compute average treatment effect (ATE)
    if proba_treatment is not None and proba_control is not None:
        # Uplift = probability of churn in control - probability of churn in treatment
        # Positive uplift means treatment reduces churn (good!)
        # Note: we compute on pooled test set by predicting with both models
        if len(X_test) > 0:
            pred_proba_treatment_all = treatment_model.predict_proba(X_test)[:, 1]
            pred_proba_control_all = control_model.predict_proba(X_test)[:, 1]

            # Uplift is the difference in churn probability: control - treatment
            uplift = pred_proba_control_all - pred_proba_treatment_all
            ate = np.mean(uplift)
            ate_std = np.std(uplift)

            results["average_treatment_effect"] = float(ate)
            results["ate_std"] = float(ate_std)
            results["uplift_interpretation"] = (
                "Positive ATE means treatment (outreach) reduces churn probability on average"
            )
            print(
                f"\nAverage Treatment Effect (ATE) on churn probability: {ate:.4f} ± {ate_std:.4f}"
            )
            print(results["uplift_interpretation"])

    return results, treatment_model, control_model


def train_s_learner_uplift(
    X_train, y_train, X_test, y_test, treatment_train, treatment_test
):
    """
    S-learner (meta-learner): train a single model that receives the treatment
    indicator as an input feature. Estimate uplift by predicting outcomes with
    treatment=1 and treatment=0 for the same feature vectors.

    Returns: results dict, trained_model
    """
    print("\n--- Training S-learner (single model with treatment as feature) ---")

    # Add treatment column to feature matrices
    X_train_aug = X_train.copy().reset_index(drop=True)
    X_train_aug["outreach"] = treatment_train.reset_index(drop=True)

    X_test_aug = X_test.copy().reset_index(drop=True)
    X_test_aug["outreach"] = treatment_test.reset_index(drop=True)

    # Train model on augmented features
    scale_pos_weight = (
        (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
    )
    model = XGBClassifier(
        eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train_aug, y_train)

    # Compute predictions for test set under both treatment settings
    X_test_treated = X_test.copy().reset_index(drop=True)
    X_test_treated["outreach"] = 1
    X_test_control = X_test.copy().reset_index(drop=True)
    X_test_control["outreach"] = 0

    proba_treated = model.predict_proba(X_test_treated)[:, 1]
    proba_control = model.predict_proba(X_test_control)[:, 1]

    # Uplift is control_prob - treated_prob (positive => treatment reduces churn)
    uplift = proba_control - proba_treated
    ate = float(np.mean(uplift))
    ate_std = float(np.std(uplift))

    results = {
        "average_treatment_effect": ate,
        "ate_std": ate_std,
        "uplift_interpretation": "Positive ATE means treatment (outreach) reduces churn probability on average",
        "n_test": len(X_test),
    }

    # Evaluate AUC on test set (using predictions corresponding to actual treatment assignment)
    # pick probabilities according to observed treatment in test set
    observed_proba = np.where(
        treatment_test.reset_index(drop=True).values == 1, proba_treated, proba_control
    )
    try:
        test_auc = roc_auc_score(y_test, observed_proba)
    except Exception:
        test_auc = None
    results["test_auc"] = float(test_auc) if test_auc is not None else None

    print(f"S-learner test AUC (observed treatment assignment) = {results['test_auc']}")
    print(
        f"Average Treatment Effect (ATE) on churn probability: {ate:.4f} ± {ate_std:.4f}"
    )

    return results, model


def run():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_dir = os.path.join(repo_root, "data", "train")
    test_dir = os.path.join(repo_root, "data", "test")

    print("=" * 70)
    print("UPLIFT MODEL TRAINING")
    print("=" * 70)
    print("\nAggregating train features...")
    train = aggregate_features(train_dir, config)
    print("Aggregating test features...")
    test = aggregate_features(test_dir, config)

    feature_cols = [
        c
        for c in train.columns
        if c not in ["member_id", "signup_date", "churn", "outreach"]
    ]

    X_train = train[feature_cols]
    y_train = train["churn"].astype(int)
    treatment_train = train["outreach"].astype(int)

    X_test = test[feature_cols]
    y_test = test["churn"].astype(int)
    treatment_test = test["outreach"].astype(int)

    # Combine for two-model approach
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = pd.concat([y_train, y_test], ignore_index=True)
    treatment_combined = pd.concat([treatment_train, treatment_test], ignore_index=True)

    print(f"\nTraining data: {len(X_train)} samples; test data: {len(X_test)} samples")
    print(
        f"Treatment in train: {treatment_train.sum()} / {len(treatment_train)} ({100*treatment_train.mean():.1f}%)"
    )
    print(
        f"Treatment in test: {treatment_test.sum()} / {len(treatment_test)} ({100*treatment_test.mean():.1f}%)"
    )

    # Choose uplift method (two-model vs meta-learner S-learner)
    uplift_method = config.get("uplift_method", "two_model")
    models_dir = os.path.join(repo_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    outputs_dir = os.path.join(repo_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    if uplift_method == "s_learner":
        print("\n" + "=" * 70)
        print(
            "S-LEARNER (meta-learner): training single model with treatment as feature"
        )
        print("=" * 70)

        uplift_results, s_model = train_s_learner_uplift(
            X_train, y_train, X_test, y_test, treatment_train, treatment_test
        )

        # Save single model
        s_model_path = os.path.join(
            models_dir, f"uplift_s_learner_model_{timestamp}.joblib"
        )
        joblib.dump(s_model, s_model_path)
        print(f"\nSaved S-learner model to: {s_model_path}")

        report = {
            "model_type": "s_learner",
            "timestamp": timestamp,
            "training_data": {
                "total_train_samples": len(X_train),
                "total_test_samples": len(X_test),
                "treatment_in_train_pct": float(100 * treatment_train.mean()),
                "treatment_in_test_pct": float(100 * treatment_test.mean()),
            },
            "results": uplift_results,
            "model_path": s_model_path,
        }

        report_path = os.path.join(
            outputs_dir, f"uplift_report_s_learner_{timestamp}.json"
        )
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nSaved uplift report to: {report_path}")

    else:
        # default: two-model approach
        print("\n" + "=" * 70)
        print("TWO-MODEL APPROACH: Training separate models for treatment and control")
        print("=" * 70)

        uplift_results, treatment_model, control_model = train_two_model_uplift(
            X_train,
            y_train,
            X_test,
            y_test,
            pd.concat([treatment_train, treatment_test], ignore_index=True),
        )

        # Save models
        if treatment_model is not None:
            treatment_model_path = os.path.join(
                models_dir, f"uplift_treatment_model_{timestamp}.joblib"
            )
            joblib.dump(treatment_model, treatment_model_path)
            print(f"\nSaved treatment model to: {treatment_model_path}")

        if control_model is not None:
            control_model_path = os.path.join(
                models_dir, f"uplift_control_model_{timestamp}.joblib"
            )
            joblib.dump(control_model, control_model_path)
            print(f"Saved control model to: {control_model_path}")

        report = {
            "model_type": "two_model",
            "timestamp": timestamp,
            "training_data": {
                "total_train_samples": len(X_train),
                "total_test_samples": len(X_test),
                "treatment_in_train_pct": float(100 * treatment_train.mean()),
                "treatment_in_test_pct": float(100 * treatment_test.mean()),
            },
            "results": uplift_results,
        }

        report_path = os.path.join(
            outputs_dir, f"uplift_report_two_model_{timestamp}.json"
        )
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nSaved uplift report to: {report_path}")

        # Generate feature importance comparison
        if treatment_model is not None and control_model is not None:
            print("\n" + "=" * 70)
            print("FEATURE IMPORTANCE COMPARISON")
            print("=" * 70)

            importance_df = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "treatment_importance": treatment_model.feature_importances_,
                    "control_importance": control_model.feature_importances_,
                }
            )
            importance_df["importance_diff"] = (
                importance_df["treatment_importance"]
                - importance_df["control_importance"]
            )
            importance_df = importance_df.sort_values(
                "importance_diff", ascending=False
            )

            print("\nTop features with largest importance differences:")
            print(importance_df.head(10).to_string(index=False))

            importance_path = os.path.join(
                outputs_dir, f"uplift_feature_importance_{timestamp}.csv"
            )
            importance_df.to_csv(importance_path, index=False)
            print(f"\nSaved feature importance to: {importance_path}")

    print("\n" + "=" * 70)
    print("UPLIFT TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run()
