import pandas as pd
from IPython.display import display


def get_web_feats(web_path, config):
    web = pd.read_csv(web_path)
    if config.get("web_feats").get("use_custom_features") is not True:
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

        result = (
            pd.concat([total, domain_counts, category_counts], axis=1)
            .fillna(0)
            .astype(int)
            .reset_index()
        )

    return result
