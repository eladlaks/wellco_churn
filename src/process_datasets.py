import os
import math
import numpy as np
import pandas as pd
from IPython.display import display


def get_web_feats(web_path, config):
    web = pd.read_csv(web_path)
    if config.get("web_feats").get("use_custom_features_web") is not True:
        result = web.groupby("member_id").size().rename("web_visit_count")
    else:
        web["url_category"] = web["url"].str.split("/").str[3]
        web["domain"] = web["url"].str.split("/").str[2]
        web.drop(columns=["description"], inplace=True)
        web["category"] = web["url_category"] + "_" + web["title"]
        web.drop(columns=["title", "url_category"], inplace=True)
        # aggregate per-member totals and counts by domain and category
        total = web.groupby("member_id").size().rename("total_visits")

        domain_counts = (
            web.groupby(["member_id", "domain"]).size().unstack(fill_value=0)
        )
        domain_counts.columns = [f"domain_{c}" for c in domain_counts.columns]

        category_counts = (
            web.groupby(["member_id", "category"]).size().unstack(fill_value=0)
        )
        category_counts.columns = [f"category_{c}" for c in category_counts.columns]
        if config["web_feats"]["use_domain"]:
            result = (
                pd.concat([total, domain_counts, category_counts], axis=1)
                .fillna(0)
                .astype(int)
                .reset_index()
            )
        else:
            result = (
                pd.concat([total, category_counts], axis=1)
                .fillna(0)
                .astype(int)
                .reset_index()
            )

    return result


def process_datasets(X_train, X_test):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    import pandas as pd

    # One-hot encode categorical columns (low-cardinality) and standardize numeric columns.

    # Copy to avoid mutating original
    X_tr = X_train.copy()
    X_te = X_test.copy()

    # Drop high-cardinality object/date columns (these are already represented by recency/count features)
    date_like = {"last_app_session", "last_web_visit", "last_claim_date"}
    obj_cols = X_tr.select_dtypes(include=["object"]).columns.tolist()
    drop_cols = [c for c in obj_cols if (c in date_like) or (X_tr[c].nunique() > 50)]
    if drop_cols:
        X_tr = X_tr.drop(columns=drop_cols)
        X_te = X_te.drop(columns=drop_cols)

    # Determine categorical (bool + low-cardinality objects) and numeric columns
    categorical_cols = X_tr.select_dtypes(
        include=["bool", "category"]
    ).columns.tolist() + [
        c
        for c in X_tr.select_dtypes(include=["object"]).columns.tolist()
        if X_tr[c].nunique() <= 50
    ]
    numeric_cols = X_tr.select_dtypes(include=["number"]).columns.tolist()

    # Build preprocessing pipeline
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    # Fit on training data and transform both train/test
    preprocessor.fit(X_tr)
    X_train_processed = preprocessor.transform(X_tr)
    X_test_processed = preprocessor.transform(X_te)

    # Get feature names and convert to DataFrame for convenience
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback for older sklearn: construct names manually
        num_names = numeric_cols
        cat_ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        cat_feature_names = []
        if hasattr(cat_ohe, "get_feature_names_out"):
            cat_feature_names = cat_ohe.get_feature_names_out(categorical_cols).tolist()
        else:
            # best-effort fallback
            for c in categorical_cols:
                uniques = X_tr[c].astype(str).unique()[:50]
                cat_feature_names += [f"{c}__{u}" for u in uniques]
        feature_names = [f"num__{n}" for n in num_names] + cat_feature_names

    X_train_processed = pd.DataFrame(
        X_train_processed, columns=feature_names, index=X_tr.index
    )
    X_test_processed = pd.DataFrame(
        X_test_processed, columns=feature_names, index=X_te.index
    )

    print("Preprocessing complete.")
    print(f"X_train -> {X_train.shape} -> {X_train_processed.shape}")
    print(f"X_test  -> {X_test.shape} -> {X_test_processed.shape}")
    return X_train_processed, X_test_processed


def create_data(repo_root, config):
    train_dir = os.path.join(repo_root, "data", "train")
    test_dir = os.path.join(repo_root, "data", "test")

    print("Aggregating train features...")
    train = aggregate_features(train_dir, config)
    print("Aggregating test features...")
    test = aggregate_features(test_dir, config)
    columns_to_drop = ["member_id", "signup_date", "churn"]
    feature_cols = [c for c in train.columns if c not in columns_to_drop]

    X_train = train[feature_cols]
    y_train = train["churn"].astype(int)
    X_test = test[feature_cols]
    y_test = test["churn"].astype(int)

    print(f"Training data: {len(X_train)} samples; test data: {len(X_test)} samples")
    X_train_processed, X_test_processed = process_datasets(X_train, X_test)
    X_train_processed.to_csv("X_train_processed.csv", index=False)
    X_test_processed.to_csv("X_test_processed.csv", index=False)
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    return X_train_processed, y_train, X_test_processed, y_test, train, test


def pick(path_dir, name):
    p1 = os.path.join(path_dir, name)
    p2 = os.path.join(path_dir, f"test_{name}")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return p1  # default (may not exist)


def aggregate_features(data_dir, config):
    """Helper to accept either canonical filenames or ones prefixed with 'test_'"""

    app_path = pick(data_dir, "app_usage.csv")
    web_path = pick(data_dir, "web_visits.csv")
    claims_path = pick(data_dir, "claims.csv")
    labels_path = pick(data_dir, "churn_labels.csv")

    # Read files (some files may be missing in tiny test fixtures; handle gracefully)
    app = pd.DataFrame()
    web = pd.DataFrame()
    claims = pd.DataFrame()

    if os.path.exists(app_path):
        app = pd.read_csv(app_path)
    if os.path.exists(web_path):
        web = pd.read_csv(web_path)
    if os.path.exists(claims_path):
        claims = pd.read_csv(claims_path)

    # Helper: detect a datetime-like column and return parsed Series
    def _find_and_parse_date(df, candidates):
        for c in candidates:
            if c in df.columns:
                try:
                    return pd.to_datetime(df[c], errors="coerce")
                except Exception:
                    continue
        return None

    # Determine observation window from available timestamp/date columns
    all_dates = []
    if not app.empty:
        ts = _find_and_parse_date(
            app, ["timestamp", "event_time", "date", "session_start", "session_end"]
        )
        if ts is not None:
            all_dates.append(ts.dropna())
    if not web.empty:
        ts = _find_and_parse_date(
            web, ["timestamp", "event_time", "date", "visit_time"]
        )
        if ts is not None:
            all_dates.append(ts.dropna())
    if not claims.empty:
        ts = _find_and_parse_date(
            claims, ["diagnosis_date", "claim_date", "service_date", "date"]
        )
        if ts is not None:
            all_dates.append(ts.dropna())

    if len(all_dates) > 0 and any(len(s) > 0 for s in all_dates):
        combined = pd.concat(all_dates)
        obs_start = combined.min()
        obs_end = combined.max()
    else:
        obs_start = None
        obs_end = None

    # ------------------ APP features ------------------
    if not app.empty:
        # parse timestamp
        app_ts = _find_and_parse_date(
            app, ["timestamp", "event_time", "date", "session_start"]
        )
        if app_ts is not None:
            app["timestamp_parsed"] = app_ts
        # session count
        app_feats = app.groupby("member_id").size().rename("session_count").to_frame()
        # last session date and recency
        if "timestamp_parsed" in app.columns:
            last = (
                app.groupby("member_id")["timestamp_parsed"]
                .max()
                .rename("last_app_session")
            )
            app_feats = app_feats.join(last)
            if obs_end is not None:
                app_feats["recency_last_app_days"] = (
                    obs_end - app_feats["last_app_session"]
                ).dt.days
            else:
                app_feats["recency_last_app_days"] = 0
        else:
            app_feats["recency_last_app_days"] = np.nan

        # average session duration if a duration-like column exists
        duration_cols = [
            c for c in app.columns if "duration" in c.lower() or "length" in c.lower()
        ]
        if len(duration_cols) > 0:
            dcol = duration_cols[0]
            app_feats["avg_session_duration"] = app.groupby("member_id")[dcol].mean()
        else:
            app_feats["avg_session_duration"] = np.nan

        # proportion of active days in observation window (app only)
        if (
            "timestamp_parsed" in app.columns
            and obs_start is not None
            and obs_end is not None
        ):
            days_window = max(1, (obs_end - obs_start).days + 1)
            active_days = (
                app.groupby("member_id")["timestamp_parsed"]
                .apply(lambda s: s.dt.normalize().nunique())
                .rename("app_active_days")
            )
            app_feats = app_feats.join(active_days)
            app_feats["prop_active_days_app"] = (
                app_feats["app_active_days"] / days_window
            )
        else:
            app_feats["prop_active_days_app"] = 0.0
    else:
        app_feats = pd.DataFrame(
            columns=[
                "session_count",
                "recency_last_app_days",
                "avg_session_duration",
                "prop_active_days_app",
            ]
        )

    # ------------------ WEB features ------------------
    # reuse existing helper when possible, but compute recency, entropy, critical page counts, and active-day proportion
    if not web.empty:
        try:
            web_feats_raw = get_web_feats(web_path, config)
            # normalize
            if isinstance(web_feats_raw, pd.DataFrame):
                if "member_id" in web_feats_raw.columns:
                    web_feats_raw = web_feats_raw.set_index("member_id")
            if isinstance(web_feats_raw, pd.Series):
                web_feats_raw = web_feats_raw.to_frame()
        except Exception:
            web_feats_raw = (
                web.groupby("member_id").size().rename("web_visit_count").to_frame()
            )

        web_ts = _find_and_parse_date(
            web, ["timestamp", "event_time", "date", "visit_time"]
        )
        if web_ts is not None:
            web["timestamp_parsed"] = web_ts
            last_web = (
                web.groupby("member_id")["timestamp_parsed"]
                .max()
                .rename("last_web_visit")
            )
        else:
            last_web = None

        # entropy of urls or titles
        if "url" in web.columns:
            # compute entropy per member
            def _entropy(s):
                counts = s.value_counts()
                p = counts / counts.sum()
                return -(p * np.log2(p)).sum()

            ent = web.groupby("member_id")["url"].apply(_entropy).rename("url_entropy")
        elif "title" in web.columns:
            ent = (
                web.groupby("member_id")["title"]
                .apply(_entropy)
                .rename("title_entropy")
            )
        else:
            ent = pd.Series(dtype=float, name="url_entropy")

        # critical page counts by keyword
        keywords = [
            "support",
            "cancel",
            "cancellation",
            "pricing",
            "cost",
            "billing",
            "payment",
            "refund",
        ]

        def contains_keyword(x):
            x = (str(x) if not pd.isna(x) else "").lower()
            return any(k in x for k in keywords)

        # search in url, title, description if present
        matches = pd.Series(False, index=web.index)
        for col in ["url", "title", "description"]:
            if col in web.columns:
                matches = matches | web[col].apply(contains_keyword)

        critical_counts = (
            web.loc[matches].groupby("member_id").size().rename("critical_page_visits")
        )

        # proportion of active days on web
        if (
            "timestamp_parsed" in web.columns
            and obs_start is not None
            and obs_end is not None
        ):
            days_window = max(1, (obs_end - obs_start).days + 1)
            web_active_days = (
                web.groupby("member_id")["timestamp_parsed"]
                .apply(lambda s: s.dt.normalize().nunique())
                .rename("web_active_days")
            )
            prop_web = (web_active_days / days_window).rename("prop_active_days_web")
        else:
            prop_web = pd.Series(dtype=float, name="prop_active_days_web")

        web_feats = (
            web_feats_raw.join(ent, how="left")
            .join(critical_counts, how="left")
            .join(prop_web, how="left")
        )
        if last_web is not None:
            web_feats = web_feats.join(last_web)
            if obs_end is not None:
                web_feats["recency_last_web_days"] = (
                    obs_end - web_feats["last_web_visit"]
                ).dt.days
            else:
                web_feats["recency_last_web_days"] = 0
        web_feats = web_feats.fillna(0)
    else:
        web_feats = pd.DataFrame(
            columns=[
                "web_visit_count",
                "url_entropy",
                "critical_page_visits",
                "prop_active_days_web",
                "recency_last_web_days",
            ]
        )

    # ------------------ CLAIMS features ------------------
    if not claims.empty:
        claims_count = (
            claims.groupby("member_id").size().rename("claims_count").to_frame()
        )
        # recency of last diagnosis/claim
        claim_ts = _find_and_parse_date(
            claims, ["diagnosis_date", "claim_date", "service_date", "date"]
        )
        if claim_ts is not None:
            claims["date_parsed"] = claim_ts
            last_claim = (
                claims.groupby("member_id")["date_parsed"]
                .max()
                .rename("last_claim_date")
            )
            claims_count = claims_count.join(last_claim)
            if obs_end is not None:
                claims_count["recency_last_claim_days"] = (
                    obs_end - claims_count["last_claim_date"]
                ).dt.days
            else:
                claims_count["recency_last_claim_days"] = 0
        else:
            claims_count["recency_last_claim_days"] = np.nan

        # unique ICD-10 codes count
        if "icd_code" in claims.columns:
            unique_icd = (
                claims.groupby("member_id")["icd_code"]
                .nunique()
                .rename("unique_icd_count")
            )
            claims_count = claims_count.join(unique_icd)
            # flags for some ICD codes of interest (existing)
            codes = ["E11.9", "I10", "Z71.3"]
            for code in codes:
                col = f"has_{code.replace('.', '_')}"
                flag = claims["icd_code"].fillna("").str.startswith(code)
                flag_series = claims.loc[flag].groupby("member_id").size().rename(col)
                flag_series = (flag_series >= 1).astype(int)
                claims_count = claims_count.join(flag_series)
        else:
            claims_count["unique_icd_count"] = 0

        # ICD prefix groups (first 3 chars) - top prefixes
        # if "icd_code" in claims.columns:
        #     claims["icd_prefix"] = claims["icd_code"].fillna("").str.slice(0, 3)
        #     prefix_counts = (
        #         claims.groupby(["member_id", "icd_prefix"]).size().unstack(fill_value=0)
        #     )
        #     # keep up to 5 prefixes as features
        #     if prefix_counts.shape[1] > 0:
        #         top_prefixes = (
        #             prefix_counts.sum(axis=0)
        #             .sort_values(ascending=False)
        #             .head(5)
        #             .index.tolist()
        #         )
        #         prefix_counts = prefix_counts[top_prefixes]
        #         prefix_counts.columns = [
        #             f"icd_prefix_{c}" for c in prefix_counts.columns
        #         ]
        #         claims_count = claims_count.join(prefix_counts)

        claims_feats = claims_count.fillna(0)
    else:
        claims_feats = pd.DataFrame()

    # ------------------ LABELS / MEMBER features ------------------
    labels = (
        pd.read_csv(labels_path, parse_dates=["signup_date"])
        if os.path.exists(labels_path)
        else pd.DataFrame()
    )
    if not labels.empty:
        labels = labels.set_index("member_id")
        # tenure relative to observation end
        if obs_end is not None:
            labels["member_tenure_days"] = (obs_end - labels["signup_date"]).dt.days
        else:
            labels["member_tenure_days"] = (
                pd.to_datetime("today") - labels["signup_date"]
            ).dt.days

        # tenure bins
        def _tenure_bin(d):
            if pd.isna(d):
                return "unknown"
            if d <= 30:
                return "new"
            if d <= 365:
                return "mid"
            return "long"

        labels["tenure_bin"] = labels["member_tenure_days"].apply(_tenure_bin)
        # one-hot encode the tenure bin (simple)
        tenure_dummies = pd.get_dummies(labels["tenure_bin"], prefix="tenure")
        labels = labels.join(tenure_dummies)
    else:
        labels = pd.DataFrame()

    # ------------------ Merge all feature frames ------------------
    # ensure indexes are member_id
    def _ensure_df(df_like):
        if isinstance(df_like, pd.Series):
            return df_like.to_frame()
        return df_like

    app_feats = _ensure_df(app_feats)
    web_feats = _ensure_df(web_feats)
    claims_feats = _ensure_df(claims_feats)

    feats = pd.concat([app_feats, web_feats, claims_feats], axis=1)
    feats = feats.fillna(0)

    # join with labels
    if not labels.empty:
        df = labels.join(feats, how="left")
        df = df.fillna(0)
        return df.reset_index()
    else:
        # return features only, reset index
        feats = feats.reset_index()
        if "member_id" not in feats.columns and feats.index.name is None:
            feats["member_id"] = feats.index
        return feats.reset_index(drop=True)
