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
