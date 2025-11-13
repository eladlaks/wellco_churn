import os
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
    return X_train, y_train, X_test, y_test, train, test


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
    # ensure app timestamps are parsed as datetimes so min() returns a Timestamp, not a string
    if not app.empty:
        first_date_window = pd.to_datetime(app["timestamp"]).min()
    else:
        first_date_window = None
    # Read labels and parse signup_date as datetime so we can do vectorized datetime arithmetic
    labels = pd.read_csv(labels_path, parse_dates=["signup_date"])
    if first_date_window is not None:
        # compute difference in days between the first event in the app data and signup_date
        labels["days_already_on_app"] = (
            first_date_window - labels["signup_date"]
        ).dt.days
        labels["days_already_on_app"] = (
            labels["days_already_on_app"].fillna(0).astype(int)
        )
    else:
        labels["days_already_on_app"] = 0
    labels = labels.set_index("member_id")

    # Merge
    df = labels.join(feats, how="left")
    df = df.fillna(0)

    return df.reset_index()
