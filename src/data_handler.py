import pandas as pd
from functools import reduce
from typing import Tuple, List


class DataHandler:
    """
    Handles data preprocessing and feature engineering for WellCo churn prediction.

    This class processes app usage, web visits, claims, and churn labels data
    to create a comprehensive feature set for churn prediction modeling.
    """

    # Constants
    MID_DATE = pd.Timestamp("2025-07-07")
    REFERENCE_DATE = pd.Timestamp("2025-07-14")
    DEFAULT_DATE = pd.Timestamp("2025-07-01")

    PRIORITY_ICD_CODES = [
        "Z71.3",
        "J00",
        "M54.5",
        "I10",
        "E11.9",
        "K21.9",
        "R51",
        "A09",
        "B34.9",
        "H10.9",
    ]

    def __init__(self, day_first_web: bool = False):
        """
        Initialize the DataHandler.

        Args:
            day_first_web: Whether to parse web visit timestamps with day-first format.
        """
        self.day_first_web = day_first_web

    def process_churn_labels(self, churn_labels: pd.DataFrame) -> pd.DataFrame:
        """
        Process churn labels to create tenure and target variables.

        Args:
            churn_labels: DataFrame with member_id, signup_date, churn, outreach columns.

        Returns:
            Processed DataFrame with tenure_days, treatment, and y columns.
        """
        df = churn_labels.copy()
        df["signup_date"] = pd.to_datetime(df["signup_date"])
        df["tenure_days"] = (self.REFERENCE_DATE - df["signup_date"]).dt.days
        df["treatment"] = df["outreach"]
        df["y"] = df["churn"]
        return df

    def calculate_usage_trend(self, group: pd.DataFrame) -> pd.Series:
        """
        Calculate early vs late usage trend for app usage data.

        Args:
            group: DataFrame group with timestamp column.

        Returns:
            Series with early_usage, late_usage, and usage_trend.
        """
        early = (group["timestamp"] <= self.MID_DATE).sum()
        late = (group["timestamp"] > self.MID_DATE).sum()
        return pd.Series(
            {"early_usage": early, "late_usage": late, "usage_trend": late - early}
        )

    def process_app_usage(self, app_usage: pd.DataFrame) -> pd.DataFrame:
        """
        Process app usage data to create aggregated features and trends.

        Args:
            app_usage: DataFrame with member_id and timestamp columns.

        Returns:
            DataFrame with app usage features per member.
        """
        df = app_usage.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Aggregate basic usage statistics
        app_usage_agg = (
            df.groupby("member_id")
            .agg(
                app_sessions=("timestamp", "count"),
                active_days=("timestamp", lambda x: x.dt.date.nunique()),
                first_session=("timestamp", "min"),
                last_session=("timestamp", "max"),
            )
            .reset_index()
        )

        # Calculate usage trends
        app_trend = (
            df.groupby("member_id").apply(self.calculate_usage_trend).reset_index()
        )

        # Merge aggregations and trends
        app_features = pd.merge(app_usage_agg, app_trend, on="member_id")
        return app_features

    def calculate_web_trend(self, group: pd.DataFrame) -> pd.Series:
        """
        Calculate early vs late visit trend for web visits data.

        Args:
            group: DataFrame group with timestamp column.

        Returns:
            Series with early_visits, late_visits, and visit_trend.
        """
        early = (group["timestamp"] <= self.MID_DATE).sum()
        late = (group["timestamp"] > self.MID_DATE).sum()
        return pd.Series(
            {"early_visits": early, "late_visits": late, "visit_trend": late - early}
        )

    def process_web_visits(self, web_visits: pd.DataFrame) -> pd.DataFrame:
        """
        Process web visits data to create aggregated features and trends.

        Args:
            web_visits: DataFrame with member_id, timestamp, and url columns.

        Returns:
            DataFrame with web visit features per member.
        """
        df = web_visits.copy()
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce", dayfirst=self.day_first_web
        )

        # Extract URL components
        df[["domain", "category", "page"]] = df["url"].str.extract(
            r"https://([^/]+)/([^/]+)/(\d+)"
        )

        # Identify WellCo domain visits
        df["is_wellco_domain"] = df["domain"].str.contains("wellco", na=False)

        # Aggregate web visit statistics
        web_agg = (
            df.groupby("member_id")
            .agg(
                total_web_visits=("url", "count"),
                unique_domains=("domain", "nunique"),
                unique_categories=("category", "nunique"),
                unique_pages=("page", "nunique"),
                last_visit=("timestamp", "max"),
                wellco_domain_visits=("is_wellco_domain", "sum"),
            )
            .reset_index()
        )

        # Calculate WellCo domain ratio
        web_agg["ratio_wellco_domain"] = (
            web_agg["wellco_domain_visits"] / web_agg["total_web_visits"]
        )

        # Calculate visit trends
        web_trend_df = (
            df.groupby("member_id").apply(self.calculate_web_trend).reset_index()
        )

        # Merge aggregations and trends
        web_features = pd.merge(web_agg, web_trend_df, on="member_id")
        return web_features

    def calculate_claims_trend(self, group: pd.DataFrame) -> pd.Series:
        """
        Calculate early vs late claims trend.

        Args:
            group: DataFrame group with diagnosis_date column.

        Returns:
            Series with early_claims, late_claims, and claim_trend.
        """
        early = (group["diagnosis_date"] <= self.MID_DATE).sum()
        late = (group["diagnosis_date"] > self.MID_DATE).sum()
        return pd.Series(
            {"early_claims": early, "late_claims": late, "claim_trend": late - early}
        )

    def process_claims(self, claims: pd.DataFrame) -> pd.DataFrame:
        """
        Process claims data to create aggregated features with ICD code indicators.

        Args:
            claims: DataFrame with member_id, diagnosis_date, and icd_code columns.

        Returns:
            DataFrame with claims features per member.
        """
        df = claims.copy()
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"])
        df["icd_category"] = df["icd_code"].str[:3]

        # Create binary indicators for priority ICD codes
        for icd_code in self.PRIORITY_ICD_CODES:
            col_name = f"has_icd_{icd_code.replace('.', '_')}"
            df[col_name] = (df["icd_code"] == icd_code).astype(int)

        # Aggregate claims statistics
        claims_agg = (
            df.groupby("member_id")
            .agg(
                {
                    "icd_code": ["count", "nunique"],
                    "icd_category": "nunique",
                    "diagnosis_date": "max",
                    **{
                        f"has_icd_{icd.replace('.', '_')}": "max"
                        for icd in self.PRIORITY_ICD_CODES
                    },
                }
            )
            .reset_index()
        )

        # Flatten column names
        claims_agg.columns = [
            "member_id",
            "total_claims",
            "unique_icd_codes",
            "unique_icd_categories",
            "last_claim",
        ] + [f"has_icd_{icd.replace('.', '_')}" for icd in self.PRIORITY_ICD_CODES]

        # Calculate claims trends
        claims_trend_df = (
            df.groupby("member_id").apply(self.calculate_claims_trend).reset_index()
        )

        # Merge aggregations and trends
        claims_features = pd.merge(claims_agg, claims_trend_df, on="member_id")
        return claims_features

    def merge_features(self, feature_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple feature DataFrames on member_id.

        Args:
            feature_dfs: List of DataFrames to merge.

        Returns:
            Merged DataFrame with all features.
        """
        return reduce(
            lambda left, right: pd.merge(left, right, on="member_id", how="left"),
            feature_dfs,
        )

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in the feature DataFrame.

        Args:
            df: DataFrame with potential missing values.

        Returns:
            DataFrame with filled missing values.
        """
        result = df.copy()

        # Fill numeric count/trend columns with 0
        count_cols = [
            c
            for c in result.columns
            if any(k in c for k in ["count", "usage", "visits", "claims"])
        ]
        result[count_cols] = result[count_cols].fillna(0)

        # Fill date columns with default date
        date_cols = ["first_session", "last_session", "last_visit", "last_claim"]
        for col in date_cols:
            if col in result.columns:
                result[col] = pd.to_datetime(result[col])
                result[col] = result[col].fillna(self.DEFAULT_DATE)

        return result

    def extract_features(
        self, full_features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Extract X, y, and treatment from the full feature DataFrame.

        Args:
            full_features: Complete DataFrame with all features and targets.

        Returns:
            Tuple of (X, y, treatment).
        """
        # Define columns to exclude from features
        exclude_cols = [
            "signup_date",
            "churn",
            "outreach",
            "treatment",
            "y",
            "first_session",
            "last_session",
            "last_claim",
            "last_visit",
            "member_id",
        ]

        feature_cols = [c for c in full_features.columns if c not in exclude_cols]

        X = full_features[feature_cols]
        y = full_features["y"]
        treatment = full_features["treatment"]

        print("Feature matrix X shape:", X.shape)
        print("Target y distribution:\n", y.value_counts())
        print("Treatment distribution:\n", treatment.value_counts())

        return X, y, treatment

    def get_data(
        self,
        app_usage: pd.DataFrame,
        web_visits: pd.DataFrame,
        claims: pd.DataFrame,
        churn_labels: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Main pipeline to process all data sources and create feature matrix.

        Args:
            app_usage: App usage events DataFrame.
            web_visits: Web visit events DataFrame.
            claims: Claims records DataFrame.
            churn_labels: Churn labels and metadata DataFrame.

        Returns:
            Tuple of (X, y, treatment) for modeling.
        """
        # Process each data source
        churn_features = self.process_churn_labels(churn_labels)
        app_features = self.process_app_usage(app_usage)
        web_features = self.process_web_visits(web_visits)
        claims_features = self.process_claims(claims)

        # Merge all features
        full_features = self.merge_features(
            [churn_features, app_features, web_features, claims_features]
        )

        # Fill missing values
        full_features = self.fill_missing_values(full_features)

        # Extract final X, y, treatment
        return self.extract_features(full_features)


# Backward compatibility: standalone function wrapper
def get_data(app_usage, web_visits, claims, churn_labels, day_first_web=False):
    """
    Legacy function wrapper for backward compatibility.

    Args:
        app_usage: App usage events DataFrame.
        web_visits: Web visit events DataFrame.
        claims: Claims records DataFrame.
        churn_labels: Churn labels and metadata DataFrame.
        day_first_web: Whether to parse web timestamps with day-first format.

    Returns:
        Tuple of (X, y, treatment) for modeling.
    """
    handler = DataHandler(day_first_web=day_first_web)
    return handler.get_data(app_usage, web_visits, claims, churn_labels)
