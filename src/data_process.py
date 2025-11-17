import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GridSearchCV,
    train_test_split,
    cross_validate,
    cross_val_score,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from econml.dr import DRLearner
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklift.models import SoloModel

# from sklift.viz import plot_qini_curve
from sklift.datasets import fetch_megafon
from sklift.metrics import make_uplift_scorer
import os
import sys
from pathlib import Path
import yaml
from datetime import datetime
import re
from sklift.metrics import qini_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from functools import reduce
import seaborn as sns

import matplotlib.pyplot as plt


def check_column_consistency(X_train, X_test):
    """Checks if X_train and X_test have the same column names and order."""
    train_cols = X_train.columns
    test_cols = X_test.columns

    # Check column names
    if set(train_cols) == set(test_cols):
        print("✅ Column names are consistent across both datasets.")
    else:
        print("❌ Column names are NOT consistent.")
        missing_in_test = list(set(train_cols) - set(test_cols))
        missing_in_train = list(set(test_cols) - set(train_cols))
        if missing_in_test:
            print(f"   - Columns in X_train but missing in X_test: {missing_in_test}")
        if missing_in_train:
            print(f"   - Columns in X_test but missing in X_train: {missing_in_train}")

    # Check column order (only necessary if names are the same, but good practice)
    common_cols = list(set(train_cols) & set(test_cols))
    train_order = list(train_cols[train_cols.isin(common_cols)])
    test_order = list(test_cols[test_cols.isin(common_cols)])

    if train_order == test_order:
        print("✅ Column order for common columns is the same.")
    else:
        print(
            "⚠️ Column order is different for common columns. Be careful when using positional indexing."
        )

    return common_cols


def check_missing_values(X_train, X_test):
    """Checks for NaNs in both datasets and reports findings."""
    print("\n--- 2. Missing Value (NaN) Check ---")

    # Check X_train
    nan_train = X_train.isnull().sum()
    nan_train = nan_train[nan_train > 0]
    if not nan_train.empty:
        print("⚠️ NaNs found in X_train:")
        print(nan_train)
    else:
        print("✅ No NaNs found in X_train.")

    # Check X_test
    nan_test = X_test.isnull().sum()
    nan_test = nan_test[nan_test > 0]
    if not nan_test.empty:
        print("\n⚠️ NaNs found in X_test:")
        print(nan_test)
    else:
        print("\n✅ No NaNs found in X_test.")


# def analyze_data_drift(X_train, X_test, common_cols):
#     """Analyzes and plots data drift for common columns."""
#     print("\n--- 3. Data Drift Analysis and Visualization ---")

#     # 1. Separate column types
#     numerical_cols = X_train[common_cols].select_dtypes(include=np.number).columns
#     categorical_cols = X_train[common_cols].select_dtypes(include="object").columns

#     print(
#         f"Analyzing {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features for drift."
#     )

#     # --- Numerical Feature Drift (KDE Plot) ---
#     if len(numerical_cols) > 0:
#         fig_num, axes_num = plt.subplots(
#             nrows=len(numerical_cols), ncols=1, figsize=(10, 4 * len(numerical_cols))
#         )
#         if len(numerical_cols) == 1:
#             axes_num = [axes_num]  # Ensure axes is iterable if only one subplot

#         print("\n> Numerical Drift (Distribution Shift):")
#         for i, col in enumerate(numerical_cols):
#             ax = axes_num[i]

#             # Plot the density of the training data
#             sns.kdeplot(
#                 X_train[col].dropna(),
#                 label="X_train",
#                 ax=ax,
#                 fill=True,
#                 alpha=0.5,
#                 linewidth=2,
#             )
#             # Plot the density of the test data
#             sns.kdeplot(
#                 X_test[col].dropna(),
#                 label="X_test",
#                 ax=ax,
#                 fill=True,
#                 alpha=0.5,
#                 linewidth=2,
#             )

#             # Use statistical distance (e.g., difference in means) as a simple indicator
#             mean_diff = X_test[col].mean() - X_train[col].mean()

#             ax.set_title(
#                 f"Distribution Comparison for: {col}", fontsize=14, fontweight="bold"
#             )
#             ax.set_xlabel("Value")
#             ax.legend()

#             # Simple assessment of drift
#             if (
#                 abs(mean_diff) > X_train[col].std() * 0.2
#             ):  # Drift if mean shift is > 20% of std dev
#                 print(f"  ⚠️ Potential Drift in {col}: Mean shift is {mean_diff:.2f}")
#             else:
#                 print(f"  ✅ {col}: Mean shift is acceptable ({mean_diff:.2f})")

#         fig_num.tight_layout()
#         plt.show()
#         # [Image of Statistical Distribution Comparison for Data Drift]

#     # --- Categorical Feature Drift (Frequency Bar Plot) ---
#     if len(categorical_cols) > 0:
#         fig_cat, axes_cat = plt.subplots(
#             nrows=len(categorical_cols),
#             ncols=1,
#             figsize=(10, 4 * len(categorical_cols)),
#         )
#         if len(categorical_cols) == 1:
#             axes_cat = [axes_cat]  # Ensure axes is iterable

#         print("\n> Categorical Drift (Frequency Shift):")
#         for i, col in enumerate(categorical_cols):
#             ax = axes_cat[i]

#             # Calculate value counts and normalize to get frequencies
#             train_freq = (
#                 X_train[col].value_counts(normalize=True).rename("X_train").sort_index()
#             )
#             test_freq = (
#                 X_test[col].value_counts(normalize=True).rename("X_test").sort_index()
#             )

#             # Combine into a single DataFrame for easy plotting
#             df_freq = pd.concat([train_freq, test_freq], axis=1).fillna(0)
#             df_freq.plot(kind="bar", ax=ax, alpha=0.7, rot=0)

#             ax.set_title(
#                 f"Frequency Comparison for: {col}", fontsize=14, fontweight="bold"
#             )
#             ax.set_ylabel("Proportion")
#             ax.set_xlabel("Category")
#             ax.legend(title="Dataset")

#             # Simple assessment of drift using maximum absolute frequency difference
#             max_freq_diff = (df_freq["X_train"] - df_freq["X_test"]).abs().max()
#             if (
#                 max_freq_diff > 0.1
#             ):  # Drift if any category shift is > 10 percentage points
#                 print(
#                     f"  ⚠️ Potential Drift in {col}: Max freq difference {max_freq_diff:.2f}"
#                 )
#             else:
#                 print(
#                     f"  ✅ {col}: Frequency shift is acceptable ({max_freq_diff:.2f})"
#                 )

#         fig_cat.tight_layout()
#         plt.show()
#         #


def dataset_health_check(X_train, X_test):
    """Performs all required checks."""
    print("=========================================")
    print("     DATASET HEALTH CHECK STARTING     ")
    print("=========================================")

    # 1. Check Column Consistency
    print("\n--- 1. Column Consistency Check ---")
    common_cols = check_column_consistency(X_train, X_test)

    # 2. Check Missing Values
    check_missing_values(X_train, X_test)

    # 3. Analyze Data Drift (only on common columns)
    if common_cols:
        analyze_data_drift(X_train, X_test, common_cols)
    else:
        print("\n--- 3. Data Drift Analysis Skipped ---")
        print("Cannot check for drift because there are no common columns.")


def check_column_consistency(X_train, X_test):
    """Checks if X_train and X_test have the same column names and order."""
    train_cols = X_train.columns
    test_cols = X_test.columns

    # Check column names
    if set(train_cols) == set(test_cols):
        print("✅ Column names are consistent across both datasets.")
    else:
        print("❌ Column names are NOT consistent.")
        missing_in_test = list(set(train_cols) - set(test_cols))
        missing_in_train = list(set(test_cols) - set(train_cols))
        if missing_in_test:
            print(f"   - Columns in X_train but missing in X_test: {missing_in_test}")
        if missing_in_train:
            print(f"   - Columns in X_test but missing in X_train: {missing_in_train}")

    # Check column order (only necessary if names are the same, but good practice)
    common_cols = list(set(train_cols) & set(test_cols))
    train_order = list(train_cols[train_cols.isin(common_cols)])
    test_order = list(test_cols[test_cols.isin(common_cols)])

    if train_order == test_order:
        print("✅ Column order for common columns is the same.")
    else:
        print(
            "⚠️ Column order is different for common columns. Be careful when using positional indexing."
        )

    return common_cols


def check_missing_values(X_train, X_test):
    """Checks for NaNs in both datasets and reports findings."""
    print("\n--- 2. Missing Value (NaN) Check ---")

    # Check X_train
    nan_train = X_train.isnull().sum()
    nan_train = nan_train[nan_train > 0]
    if not nan_train.empty:
        print("⚠️ NaNs found in X_train:")
        print(nan_train)
    else:
        print("✅ No NaNs found in X_train.")

    # Check X_test
    nan_test = X_test.isnull().sum()
    nan_test = nan_test[nan_test > 0]
    if not nan_test.empty:
        print("\n⚠️ NaNs found in X_test:")
        print(nan_test)
    else:
        print("\n✅ No NaNs found in X_test.")


def analyze_data_drift(X_train, X_test, common_cols):
    """Analyzes and plots data drift for common columns."""
    print("\n--- 3. Data Drift Analysis and Visualization ---")

    # 1. Separate column types
    numerical_cols = X_train[common_cols].select_dtypes(include=np.number).columns
    categorical_cols = X_train[common_cols].select_dtypes(include="object").columns

    print(
        f"Analyzing {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features for drift."
    )

    # --- Numerical Feature Drift (KDE Plot) ---
    if len(numerical_cols) > 0:
        fig_num, axes_num = plt.subplots(
            nrows=len(numerical_cols), ncols=1, figsize=(10, 4 * len(numerical_cols))
        )
        if len(numerical_cols) == 1:
            axes_num = [axes_num]  # Ensure axes is iterable if only one subplot

        print("\n> Numerical Drift (Distribution Shift):")
        for i, col in enumerate(numerical_cols):
            ax = axes_num[i]

            # Plot the density of the training data
            sns.kdeplot(
                X_train[col].dropna(),
                label="X_train",
                ax=ax,
                fill=True,
                alpha=0.5,
                linewidth=2,
            )
            # Plot the density of the test data
            sns.kdeplot(
                X_test[col].dropna(),
                label="X_test",
                ax=ax,
                fill=True,
                alpha=0.5,
                linewidth=2,
            )

            # Use statistical distance (e.g., difference in means) as a simple indicator
            mean_diff = X_test[col].mean() - X_train[col].mean()

            ax.set_title(
                f"Distribution Comparison for: {col}", fontsize=14, fontweight="bold"
            )
            ax.set_xlabel("Value")
            ax.legend()

            # Simple assessment of drift
            if (
                abs(mean_diff) > X_train[col].std() * 0.2
            ):  # Drift if mean shift is > 20% of std dev
                print(f"  ⚠️ Potential Drift in {col}: Mean shift is {mean_diff:.2f}")
            else:
                print(f"  ✅ {col}: Mean shift is acceptable ({mean_diff:.2f})")

        fig_num.tight_layout()
        plt.show()
        # [Image of Statistical Distribution Comparison for Data Drift]

    # --- Categorical Feature Drift (Frequency Bar Plot) ---
    if len(categorical_cols) > 0:
        fig_cat, axes_cat = plt.subplots(
            nrows=len(categorical_cols),
            ncols=1,
            figsize=(10, 4 * len(categorical_cols)),
        )
        if len(categorical_cols) == 1:
            axes_cat = [axes_cat]  # Ensure axes is iterable

        print("\n> Categorical Drift (Frequency Shift):")
        for i, col in enumerate(categorical_cols):
            ax = axes_cat[i]

            # Calculate value counts and normalize to get frequencies
            train_freq = (
                X_train[col].value_counts(normalize=True).rename("X_train").sort_index()
            )
            test_freq = (
                X_test[col].value_counts(normalize=True).rename("X_test").sort_index()
            )

            # Combine into a single DataFrame for easy plotting
            df_freq = pd.concat([train_freq, test_freq], axis=1).fillna(0)
            df_freq.plot(kind="bar", ax=ax, alpha=0.7, rot=0)

            ax.set_title(
                f"Frequency Comparison for: {col}", fontsize=14, fontweight="bold"
            )
            ax.set_ylabel("Proportion")
            ax.set_xlabel("Category")
            ax.legend(title="Dataset")

            # Simple assessment of drift using maximum absolute frequency difference
            max_freq_diff = (df_freq["X_train"] - df_freq["X_test"]).abs().max()
            if (
                max_freq_diff > 0.1
            ):  # Drift if any category shift is > 10 percentage points
                print(
                    f"  ⚠️ Potential Drift in {col}: Max freq difference {max_freq_diff:.2f}"
                )
            else:
                print(
                    f"  ✅ {col}: Frequency shift is acceptable ({max_freq_diff:.2f})"
                )

        fig_cat.tight_layout()
        plt.show()
        #


def dataset_health_check(X_train, X_test):
    """Performs all required checks."""
    print("=========================================")
    print("     DATASET HEALTH CHECK STARTING     ")
    print("=========================================")

    # 1. Check Column Consistency
    print("\n--- 1. Column Consistency Check ---")
    common_cols = check_column_consistency(X_train, X_test)

    # 2. Check Missing Values
    check_missing_values(X_train, X_test)

    # 3. Analyze Data Drift (only on common columns)
    if common_cols:
        analyze_data_drift(X_train, X_test, common_cols)
    else:
        print("\n--- 3. Data Drift Analysis Skipped ---")
        print("Cannot check for drift because there are no common columns.")


def feature_selection(corr_matrix, X_train):
    ##   Correlation Analysis: Identify Redundant Features
    # Get correlation matrix (already computed in previous cell)
    corr_abs = corr_matrix.abs()

    # 1. Find highly correlated feature pairs (threshold: 0.8)
    upper_triangle = np.triu(np.ones(corr_abs.shape), k=1).astype(bool)
    high_corr_pairs = []

    for i in range(len(corr_abs.columns)):
        for j in range(i + 1, len(corr_abs.columns)):
            if corr_abs.iloc[i, j] > 0.8:
                high_corr_pairs.append(
                    {
                        "Feature_1": corr_abs.columns[i],
                        "Feature_2": corr_abs.columns[j],
                        "Correlation": corr_abs.iloc[i, j],
                    }
                )

    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
        "Correlation", ascending=False
    )

    print("=" * 80)
    print("HIGHLY CORRELATED FEATURES (|r| > 0.8)")
    print("=" * 80)
    if len(high_corr_df) > 0:
        print(high_corr_df.head(20).to_string(index=False))
        print(f"\nTotal pairs: {len(high_corr_df)}")
    else:
        print("✅ No feature pairs with |correlation| > 0.8")

    # 2. Check correlation with target (churn) and treatment (outreach)
    target_corr = (
        corr_matrix[["churn", "outreach"]].abs().sort_values("churn", ascending=False)
    )

    print("\n" + "=" * 80)
    print("FEATURE CORRELATION WITH CHURN & OUTREACH")
    print("=" * 80)
    print(target_corr.head(20).to_string())

    # 3. Identify features to potentially drop
    # Rule: If two features correlated > 0.8, drop the one with lower correlation to churn
    features_to_drop = set()

    for _, row in high_corr_df.iterrows():
        feat1, feat2 = row["Feature_1"], row["Feature_2"]

        # Skip if already marked for dropping
        if feat1 in features_to_drop or feat2 in features_to_drop:
            continue

        # Compare correlation with churn
        if feat1 in target_corr.index and feat2 in target_corr.index:
            corr1 = target_corr.loc[feat1, "churn"]
            corr2 = target_corr.loc[feat2, "churn"]

            # Drop the feature with weaker correlation to churn
            if corr1 < corr2:
                features_to_drop.add(feat1)
            else:
                features_to_drop.add(feat2)

    print("\n" + "=" * 80)
    print("RECOMMENDED FEATURES TO DROP (Redundant + Weak Churn Correlation)")
    print("=" * 80)
    if len(features_to_drop) > 0:
        print(f"Total features to drop: {len(features_to_drop)}")
        print(sorted(features_to_drop))
    else:
        print("✅ No redundant features detected")

    # 4. Variance Inflation Factor (VIF) check for multicollinearity
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Select only numeric features (exclude target/treatment if still present)
    numeric_cols = [
        col
        for col in X_train.columns
        if col not in ["churn", "outreach"]
        and X_train[col].dtype in ["float64", "int64"]
    ]

    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_cols
    vif_data["VIF"] = [
        variance_inflation_factor(X_train[numeric_cols].fillna(0).values, i)
        for i in range(len(numeric_cols))
    ]
    vif_data = vif_data.sort_values("VIF", ascending=False)

    print("\n" + "=" * 80)
    print("VARIANCE INFLATION FACTOR (VIF) - Top 20 Features")
    print("VIF > 10 indicates severe multicollinearity")
    print("=" * 80)
    print(vif_data.head(20).to_string(index=False))

    # 5. PCA Recommendation
    print("\n" + "=" * 80)
    print("PCA vs FEATURE SELECTION RECOMMENDATION")
    print("=" * 80)

    n_features = X_train.shape[1]
    n_high_corr = len(high_corr_df)
    n_vif_high = (vif_data["VIF"] > 10).sum()

    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Total features: {n_features}")
    print(f"   Highly correlated pairs: {n_high_corr}")
    print(f"   Features with VIF > 10: {n_vif_high}")
    print(f"   Recommended to drop: {len(features_to_drop)}")

    print(f"\n💡 RECOMMENDATION:")

    return features_to_drop


def Feature_Statistics_by_Treatment_Group(
    X_train, treatment_train, X_test, treatment_test, timestamp, repo_root
):
    ## Feature Statistics by Treatment Group

    print("\n" + "=" * 80)
    print("FEATURE STATISTICS BY TREATMENT GROUP")
    print("=" * 80)

    # ---------------------------
    # 1. Calculate mean by treatment group
    # ---------------------------

    # Training set analysis
    train_features_by_treatment = X_train.groupby(treatment_train).mean()

    print("\n📊 TRAIN SET: Mean Feature Values by Treatment Group")
    print("=" * 80)
    print(train_features_by_treatment.T.head(20))  # Transpose for readability
    print(f"\n... showing first 20 of {train_features_by_treatment.shape[1]} features")

    # Test set analysis
    test_features_by_treatment = X_test.groupby(treatment_test).mean()

    print("\n📊 TEST SET: Mean Feature Values by Treatment Group")
    print("=" * 80)
    print(test_features_by_treatment.T.head(20))
    print(f"\n... showing first 20 of {test_features_by_treatment.shape[1]} features")

    # ---------------------------
    # 2. Calculate difference (Treatment - Control)
    # ---------------------------

    train_diff = train_features_by_treatment.loc[1] - train_features_by_treatment.loc[0]
    train_diff_sorted = train_diff.abs().sort_values(ascending=False)

    print("\n" + "=" * 80)
    print("TOP 20 FEATURES WITH LARGEST TREATMENT/CONTROL DIFFERENCES (TRAIN)")
    print("=" * 80)

    diff_df_train = pd.DataFrame(
        {
            "Feature": train_diff_sorted.index,
            "Difference (T-C)": train_diff[train_diff_sorted.index].values,
            "Control Mean": train_features_by_treatment.loc[
                0, train_diff_sorted.index
            ].values,
            "Treatment Mean": train_features_by_treatment.loc[
                1, train_diff_sorted.index
            ].values,
            "Abs Difference": train_diff_sorted.values,
        }
    )

    print(diff_df_train.head(20).to_string(index=False))

    # ---------------------------
    # 3. Statistical significance test (t-test)
    # ---------------------------

    from scipy.stats import ttest_ind

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TEST (Top 20 features)")
    print("=" * 80)

    significant_features = []

    for feature in train_diff_sorted.head(20).index:
        treated_values = X_train.loc[treatment_train == 1, feature]
        control_values = X_train.loc[treatment_train == 0, feature]

        # Perform t-test
        t_stat, p_value = ttest_ind(treated_values, control_values, equal_var=False)

        significant_features.append(
            {
                "Feature": feature,
                "Difference": train_diff[feature],
                "p_value": p_value,
                "Significant": "✅" if p_value < 0.05 else "❌",
            }
        )

    sig_df = pd.DataFrame(significant_features)
    sig_df = sig_df.sort_values("p_value")
    print(sig_df.to_string(index=False))

    # ---------------------------
    # 4. Visualize top differences
    # ---------------------------

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot 1: Top 15 feature differences (train)
    ax1 = axes[0, 0]
    top_15_train = train_diff_sorted.head(15)
    colors_train = [
        "#d62728" if train_diff[f] > 0 else "#2ca02c" for f in top_15_train.index
    ]

    ax1.barh(
        range(len(top_15_train)),
        [train_diff[f] for f in top_15_train.index],
        color=colors_train,
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_yticks(range(len(top_15_train)))
    ax1.set_yticklabels(top_15_train.index, fontsize=9)
    ax1.set_xlabel("Difference (Treatment - Control)", fontsize=10)
    ax1.set_title("Top 15 Feature Differences (Train Set)", fontweight="bold")
    ax1.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax1.grid(alpha=0.3, axis="x")

    # Plot 2: Treatment vs Control scatter for top feature
    ax2 = axes[0, 1]
    top_feature = train_diff_sorted.index[0]
    control_vals = X_train.loc[treatment_train == 0, top_feature]
    treatment_vals = X_train.loc[treatment_train == 1, top_feature]

    ax2.violinplot([control_vals, treatment_vals], positions=[0, 1], showmeans=True)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Control", "Treatment"])
    ax2.set_ylabel(top_feature, fontsize=10)
    ax2.set_title(f"Distribution: {top_feature[:40]}...", fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")

    # Plot 3: Heatmap of top 10 features by treatment
    ax3 = axes[1, 0]
    top_10_features = train_diff_sorted.head(10).index
    heatmap_data = train_features_by_treatment[top_10_features].T

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        ax=ax3,
        cbar_kws={"label": "Mean Value"},
        xticklabels=["Control", "Treatment"],
    )
    ax3.set_ylabel("Feature", fontsize=10)
    ax3.set_title("Top 10 Features: Treatment vs Control (Train)", fontweight="bold")

    # Plot 4: P-value distribution
    ax4 = axes[1, 1]
    p_values = sig_df["p_value"].values
    ax4.hist(p_values, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    ax4.axvline(
        0.05, color="red", linestyle="--", linewidth=2, label="p=0.05 threshold"
    )
    ax4.set_xlabel("p-value", fontsize=10)
    ax4.set_ylabel("Frequency", fontsize=10)
    ax4.set_title("P-value Distribution (t-test)", fontweight="bold")
    ax4.legend()
    ax4.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        repo_root / "outputs" / f"treatment_feature_comparison_{timestamp}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    print(
        f"\n✅ Visualization saved to: outputs/treatment_feature_comparison_{timestamp}.png"
    )

    # ---------------------------
    # 5. Summary statistics by treatment
    # ---------------------------

    print("\n" + "=" * 80)
    print("COMPLETE STATISTICS BY TREATMENT GROUP (TRAIN)")
    print("=" * 80)

    # Get full descriptive statistics
    train_stats_control = X_train[treatment_train == 0].describe()
    train_stats_treatment = X_train[treatment_train == 1].describe()

    print("\n📊 Control Group (outreach=0) - Sample Statistics:")
    print(f"   N = {(treatment_train == 0).sum()}")
    print(train_stats_control.T.head(10))

    print("\n📊 Treatment Group (outreach=1) - Sample Statistics:")
    print(f"   N = {(treatment_train == 1).sum()}")
    print(train_stats_treatment.T.head(10))

    # ---------------------------
    # 6. Export to CSV for further analysis
    # ---------------------------

    # Export training set comparison
    output_path_train = (
        repo_root / "outputs" / f"feature_comparison_train_{timestamp}.csv"
    )
    diff_df_train.to_csv(output_path_train, index=False)
    print(f"\n✅ Train comparison saved to: {output_path_train}")

    # Export test set comparison
    test_diff = test_features_by_treatment.loc[1] - test_features_by_treatment.loc[0]
    test_diff_sorted = test_diff.abs().sort_values(ascending=False)

    diff_df_test = pd.DataFrame(
        {
            "Feature": test_diff_sorted.index,
            "Difference (T-C)": test_diff[test_diff_sorted.index].values,
            "Control Mean": test_features_by_treatment.loc[
                0, test_diff_sorted.index
            ].values,
            "Treatment Mean": test_features_by_treatment.loc[
                1, test_diff_sorted.index
            ].values,
            "Abs Difference": test_diff_sorted.values,
        }
    )

    output_path_test = (
        repo_root / "outputs" / f"feature_comparison_test_{timestamp}.csv"
    )
    diff_df_test.to_csv(output_path_test, index=False)
    print(f"✅ Test comparison saved to: {output_path_test}")

    # ---------------------------
    # 7. Clinical priority features analysis
    # ---------------------------

    print("\n" + "=" * 80)
    print("CLINICAL PRIORITY FEATURES (from data/wellco_client_brief.txt)")
    print("=" * 80)

    clinical_features = [
        col
        for col in X_train.columns
        if any(
            keyword in col.lower()
            for keyword in [
                "e11.9",  # Diabetes
                "i10",  # Hypertension
                "z71.3",  # Dietary counseling
                "claims_count",
                "app_logins",
                "web_visits",
            ]
        )
    ]

    if len(clinical_features) > 0:
        clinical_comparison = pd.DataFrame(
            {
                "Feature": clinical_features,
                "Control Mean": [
                    train_features_by_treatment.loc[0, f] for f in clinical_features
                ],
                "Treatment Mean": [
                    train_features_by_treatment.loc[1, f] for f in clinical_features
                ],
                "Difference": [train_diff[f] for f in clinical_features],
            }
        )

        print(clinical_comparison.to_string(index=False))
    else:
        print("⚠️  No clinical priority features found in current feature set")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def analyze_heterogeneous_treatment_effects(X_train, y_train, treatment_train):
    ## Analyze Heterogeneous Treatment Effects by Segment

    # Prepare data with labels
    X_train_with_labels = X_train.copy()
    X_train_with_labels["churn"] = y_train.values
    X_train_with_labels["outreach"] = treatment_train.values

    # Calculate overall ATE for reference
    overall_ate = (
        y_train[treatment_train == 1].mean() - y_train[treatment_train == 0].mean()
    )

    # Features to analyze (WellCo clinical priorities)
    features_ate = X_train.columns.tolist()

    for feature in features_ate:
        if feature not in X_train.columns:
            print(f"\n⚠️  Feature '{feature}' not found in X_train, skipping...")
            continue

        # Check if feature has variation
        if X_train[feature].nunique() <= 1:
            print(f"\n⚠️  Feature '{feature}' has no variation, skipping...")
            continue

        # Handle features with few unique values (binary/categorical)
        unique_vals = X_train[feature].nunique()

        try:
            if unique_vals <= 3:
                # For binary/low-cardinality features, use actual values as segments
                X_train_with_labels[f"{feature}_level"] = X_train[feature].astype(str)
                segments = sorted(X_train_with_labels[f"{feature}_level"].unique())
            else:
                # For continuous features, create tertiles
                X_train_with_labels[f"{feature}_level"] = pd.qcut(
                    X_train[feature],
                    q=3,
                    labels=["Low", "Medium", "High"],
                    duplicates="drop",  # Handle tied values
                )
                segments = ["Low", "Medium", "High"]

            # Calculate ATE for each segment
            segment_ates = []
            for segment in segments:
                mask = X_train_with_labels[f"{feature}_level"] == segment

                # Check if we have both treated and control members in this segment
                n_treated = (mask & (treatment_train == 1)).sum()
                n_control = (mask & (treatment_train == 0)).sum()

                if n_treated == 0 or n_control == 0:
                    print(
                        f"\n⚠️  Segment '{segment}' has no treated or control members, skipping..."
                    )
                    continue

                treated_churn = X_train_with_labels.loc[
                    mask & (treatment_train == 1), "churn"
                ].mean()
                control_churn = X_train_with_labels.loc[
                    mask & (treatment_train == 0), "churn"
                ].mean()

                segment_ate = treated_churn - control_churn
                n_members = mask.sum()

                segment_ates.append(
                    {
                        "Segment": f"{segment} {feature}",
                        "ATE": segment_ate,
                        "Members": n_members,
                        "N_Treated": n_treated,
                        "N_Control": n_control,
                        "Treated_Churn": treated_churn,
                        "Control_Churn": control_churn,
                    }
                )

            if len(segment_ates) == 0:
                print(f"\n⚠️  No valid segments for '{feature}', skipping...")
                continue

            ate_df = pd.DataFrame(segment_ates)

            # Print results
            print("\n" + "=" * 80)
            print(f"TREATMENT EFFECT BY {feature.upper()} LEVEL")
            print("=" * 80)
            print(ate_df.to_string(index=False))
            print(f"\nOverall ATE (reference): {overall_ate:.4f}")
            print("=" * 80)

            # Visualize
            fig, ax = plt.subplots(figsize=(12, 6))
            x = range(len(ate_df))
            colors = ["#d62728" if ate > 0 else "#2ca02c" for ate in ate_df["ATE"]]

            bars = ax.bar(x, ate_df["ATE"], color=colors, alpha=0.7, edgecolor="black")

            # Add value labels on bars
            for i, (idx, row) in enumerate(ate_df.iterrows()):
                ax.text(
                    i,
                    row["ATE"],
                    f'{row["ATE"]:.3f}',
                    ha="center",
                    va="bottom" if row["ATE"] > 0 else "top",
                    fontsize=9,
                )

            # Reference lines
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.3)
            ax.axhline(
                y=overall_ate,
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"Overall ATE ({overall_ate:.4f})",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(ate_df["Segment"], rotation=45, ha="right")
            ax.set_ylabel(
                "Treatment Effect (Treated Churn - Control Churn)", fontsize=11
            )
            ax.set_title(
                f"Heterogeneous Treatment Effects by {feature}\n(Negative = Outreach Reduces Churn)",
                fontsize=12,
                fontweight="bold",
            )
            ax.legend(loc="best")
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            plt.show()

            # Clinical interpretation for WellCo
            print(f"\n💡 CLINICAL INSIGHT for {feature}:")
            best_segment = ate_df.loc[ate_df["ATE"].idxmin()]
            worst_segment = ate_df.loc[ate_df["ATE"].idxmax()]

            if best_segment["ATE"] < overall_ate:
                print(
                    f"   ✅ BEST: {best_segment['Segment']} shows strongest benefit (ATE={best_segment['ATE']:.4f})"
                )
                print(
                    f"      → Prioritize outreach for this segment ({best_segment['Members']} members)"
                )

            if worst_segment["ATE"] > 0:
                print(
                    f"   ⚠️  WORST: {worst_segment['Segment']} shows harm/no benefit (ATE={worst_segment['ATE']:.4f})"
                )
                print(
                    f"      → Avoid outreach for this segment ({worst_segment['Members']} members)"
                )

        except Exception as e:
            print(f"\n❌ Error processing '{feature}': {str(e)}")
            continue

    print("\n" + "=" * 80)
    print(
        "SUMMARY: Use these insights to target outreach based on uplift model predictions"
    )
    print("=" * 80)


def treatment_interation(corr_matrix, X_train, treatment_train, X_test, treatment_test):
    ## Create Treatment Interaction Features for Uplift Modeling

    print("\n" + "=" * 80)
    print("CREATING TREATMENT INTERACTION FEATURES")
    print("=" * 80)

    # 1. Find features correlated with treatment (outreach)
    # Use the correlation matrix already computed
    if "outreach" in corr_matrix.columns:
        outreach_corr = corr_matrix["outreach"].abs().sort_values(ascending=False)
        # Remove 'outreach' itself and 'churn'
        outreach_corr = outreach_corr.drop(["outreach", "churn"], errors="ignore")

        print("\nTop 20 features correlated with outreach:")
        print(outreach_corr.head(20).to_string())

        # 2. Select features with meaningful correlation (|r| > threshold)
        correlation_threshold = 0.1  # Adjust based on your data
        strong_corr_features = outreach_corr[
            outreach_corr > correlation_threshold
        ].index.tolist()

        print(
            f"\n✅ Found {len(strong_corr_features)} features with |correlation| > {correlation_threshold}"
        )
        print(f"   Features: {strong_corr_features[:10]}...")  # Show first 10

    else:
        # If correlation matrix doesn't have outreach, compute it
        print("\n⚠️  Computing correlation with outreach...")
        temp_df = X_train.copy()
        temp_df["outreach"] = treatment_train
        outreach_corr = temp_df.corr()["outreach"].abs().sort_values(ascending=False)
        outreach_corr = outreach_corr.drop("outreach", errors="ignore")

        print("\nTop 20 features correlated with outreach:")
        print(outreach_corr.head(20).to_string())

        correlation_threshold = 0.1
        strong_corr_features = outreach_corr[
            outreach_corr > correlation_threshold
        ].index.tolist()

        print(
            f"\n✅ Found {len(strong_corr_features)} features with |correlation| > {correlation_threshold}"
        )

    # 3. Create interaction features: outreach * feature
    print("\n" + "=" * 80)
    print("CREATING INTERACTION FEATURES")
    print("=" * 80)

    # Function to create interactions
    def create_treatment_interactions(X, treatment, feature_list, prefix="interact_"):
        """
        Create treatment interaction features

        Args:
            X: Feature DataFrame
            treatment: Treatment indicator (0/1)
            feature_list: List of feature names to interact with treatment
            prefix: Prefix for new feature names

        Returns:
            DataFrame with original features + interaction features
        """
        X_with_interactions = X.copy()

        for feature in feature_list:
            if feature in X.columns:
                interaction_name = f"{prefix}{feature}"
                X_with_interactions[interaction_name] = treatment * X[feature]

        return X_with_interactions

    # Limit to top N features to avoid feature explosion
    max_interactions = 15  # Adjust based on your needs
    top_interaction_features = strong_corr_features[:max_interactions]

    print(f"\nCreating interactions for top {len(top_interaction_features)} features:")
    print(f"   {top_interaction_features}")

    # Create interactions for train and test
    X_train_interact = create_treatment_interactions(
        X_train, treatment_train, top_interaction_features, prefix="outreach_x_"
    )

    X_test_interact = create_treatment_interactions(
        X_test, treatment_test, top_interaction_features, prefix="outreach_x_"
    )

    print(f"\n✅ Created {len(top_interaction_features)} interaction features")
    print(f"   Original train shape: {X_train.shape}")
    print(f"   New train shape: {X_train_interact.shape}")
    print(f"   Added features: {X_train_interact.shape[1] - X_train.shape[1]}")

    # 4. Show example of interaction features
    interaction_cols = [
        col for col in X_train_interact.columns if col.startswith("outreach_x_")
    ]
    print(f"\n📊 Interaction features created:")
    for i, col in enumerate(interaction_cols[:5], 1):
        print(f"   {i}. {col}")
    print(f"   ... and {len(interaction_cols) - 5} more")

    # 5. Verify interaction feature statistics
    print("\n" + "=" * 80)
    print("INTERACTION FEATURE STATISTICS")
    print("=" * 80)

    interaction_stats = X_train_interact[interaction_cols].describe().T
    print(interaction_stats[["mean", "std", "min", "max"]].to_string())

    # 6. Alternative: Clinical priority features (from wellco_client_brief.txt)
    # These are high-value interactions for WellCo's use case
    clinical_priority_features = [
        col
        for col in X_train.columns
        if any(
            keyword in col.lower()
            for keyword in [
                "e11.9",  # Diabetes
                "i10",  # Hypertension
                "z71.3",  # Dietary counseling
                "claims_count",
                "app_logins",
                "web_visits",
            ]
        )
    ]

    if len(clinical_priority_features) > 0:
        print("\n💡 CLINICAL INSIGHT: Creating priority interactions")
        print(f"   Priority features: {clinical_priority_features[:10]}")

        X_train_clinical = create_treatment_interactions(
            X_train_interact,
            treatment_train,
            clinical_priority_features,
            prefix="clinical_x_",
        )

        X_test_clinical = create_treatment_interactions(
            X_test_interact,
            treatment_test,
            clinical_priority_features,
            prefix="clinical_x_",
        )

        print(
            f"   Final train shape with clinical interactions: {X_train_clinical.shape}"
        )
    else:
        X_train_clinical = X_train_interact.copy()
        X_test_clinical = X_test_interact.copy()
        print("\n⚠️  No clinical priority features found in current feature set")
    return X_train_clinical, X_test_clinical
