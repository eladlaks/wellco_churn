from sklift.metrics import qini_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compare_train_test_performance(
    model, X_tr, X_te, y_tr, y_te, t_tr, t_te, model_name="Model"
):
    """Compare model performance on train vs test to detect overfitting"""

    # Predict on both sets
    uplift_train = model.predict(X_tr)
    uplift_test = model.predict(X_te)

    # Calculate Qini AUC
    qini_train = qini_auc_score(y_tr, uplift_train, t_tr)
    qini_test = qini_auc_score(y_te, uplift_test, t_te)

    # Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    from sklift.viz import plot_qini_curve

    plot_qini_curve(y_tr, uplift_train, t_tr, perfect=True, name="Train", ax=ax1)
    ax1.set_title(f"{model_name} - Train (Qini AUC: {qini_train:.4f})")

    plot_qini_curve(y_te, uplift_test, t_te, perfect=True, name="Test", ax=ax2)
    ax2.set_title(f"{model_name} - Test (Qini AUC: {qini_test:.4f})")

    plt.tight_layout()
    plt.show()

    # Calculate overfitting gap
    gap = qini_train - qini_test
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Train Qini AUC: {qini_train:.4f}")
    print(f"Test Qini AUC:  {qini_test:.4f}")
    print(f"Overfitting Gap: {gap:.4f} ({gap/qini_train*100:.1f}% relative)")

    if gap > 0.01:
        print("⚠️  WARNING: Significant overfitting detected!")
    elif gap > 0.005:
        print("⚠️  Mild overfitting detected")
    else:
        print("✅ Good generalization")
    print(f"{'='*60}\n")

    return qini_train, qini_test, gap


def calculate_qini_auuc(
    df, score_col="prioritization_score", uplift_col="churn", treatment_col="outreach"
):
    """Calculates Qini curve and Area Under the Uplift Curve (AUUC)."""

    # Sort the data by the prioritization score
    df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    N_total = len(df_sorted)
    qini_data = []

    # Calculate Total Uplift Gain for Baseline
    N_treated_total = df_sorted[treatment_col].sum()
    mean_churn_control_total = df_sorted[df_sorted[treatment_col] == 0][
        uplift_col
    ].mean()
    mean_churn_treated_total = df_sorted[df_sorted[treatment_col] == 1][
        uplift_col
    ].mean()
    qini_gain_total = (
        mean_churn_control_total - mean_churn_treated_total
    ) * N_treated_total

    for k in range(1, N_total + 1):
        df_k = df_sorted.head(k)
        y_treated = df_k[df_k[treatment_col] == 1][uplift_col]
        y_control = df_k[df_k[treatment_col] == 0][uplift_col]

        N_treated_k = len(y_treated)

        if N_treated_k > 0 and len(y_control) > 0:
            mean_churn_control = y_control.mean()
            mean_churn_treated = y_treated.mean()
            # Qini Gain: (P(Yc|Xk) - P(Yt|Xk)) * N_treated_k
            qini_gain = (mean_churn_control - mean_churn_treated) * N_treated_k
        else:
            qini_gain = 0

        qini_data.append({"k": k, "qini_gain": qini_gain})

    df_qini = pd.DataFrame(qini_data)

    # Calculate AUUC (Area between model curve and random baseline)
    df_qini["random_gain"] = df_qini["k"] / N_total * qini_gain_total
    auuc = (
        np.trapz(
            df_qini["qini_gain"].values - df_qini["random_gain"].values,
            df_qini["k"].values,
        )
        / N_total
    )

    return df_qini, auuc, qini_gain_total
