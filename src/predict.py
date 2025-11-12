import os
import argparse
import pandas as pd
import joblib


def aggregate_features(data_dir):
    # Lightweight aggregator mirroring src/train.py behavior for inference
    def pick(path_dir, name):
        p1 = os.path.join(path_dir, name)
        p2 = os.path.join(path_dir, f'test_{name}')
        if os.path.exists(p1):
            return p1
        if os.path.exists(p2):
            return p2
        return p1

    app_path = pick(data_dir, 'app_usage.csv')
    web_path = pick(data_dir, 'web_visits.csv')
    claims_path = pick(data_dir, 'claims.csv')
    labels_path = pick(data_dir, 'churn_labels.csv')

    feats = []

    if os.path.exists(app_path):
        app = pd.read_csv(app_path)
        app_feats = app.groupby('member_id').size().rename('session_count')
        feats.append(app_feats)

    if os.path.exists(web_path):
        web = pd.read_csv(web_path)
        web_feats = web.groupby('member_id').size().rename('web_visit_count')
        feats.append(web_feats)

    if os.path.exists(claims_path):
        claims = pd.read_csv(claims_path)
        claims_count = claims.groupby('member_id').size().rename('claims_count')
        codes = ['E11.9', 'I10', 'Z71.3']
        claims_df = claims_count.to_frame()
        for code in codes:
            col = f'has_{code.replace('.', '_')}'
            flag = claims['icd_code'].fillna('').str.startswith(code)
            flag_series = claims.loc[flag].groupby('member_id').size().rename(col)
            claims_df = claims_df.join(flag_series, how='left')
        claims_df = claims_df.fillna(0)
        feats.append(claims_df)

    if feats:
        feats_df = pd.concat(feats, axis=1).fillna(0)
    else:
        feats_df = pd.DataFrame()

    return feats_df.reset_index()


def main():
    parser = argparse.ArgumentParser(description='Load saved model and predict probabilities for members')
    parser.add_argument('--model', '-m', default='models/best_model.joblib', help='Path to saved model joblib')
    parser.add_argument('--features-csv', '-f', help='Path to member-level features CSV (must include member_id)')
    parser.add_argument('--data-dir', '-d', help='Path to raw data directory (will aggregate features similarly to training)')
    parser.add_argument('--output', '-o', default='outputs/predictions_from_model.csv', help='Output CSV path')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found at {args.model}")

    model = joblib.load(args.model)

    if args.features_csv:
        feats = pd.read_csv(args.features_csv)
        if 'member_id' not in feats.columns:
            raise ValueError('features CSV must contain member_id column')
        X = feats.drop(columns=['member_id'])
        member_ids = feats['member_id']
    elif args.data_dir:
        feats_df = aggregate_features(args.data_dir)
        member_ids = feats_df['member_id']
        X = feats_df.drop(columns=['member_id'])
    else:
        raise ValueError('Either --features-csv or --data-dir must be provided')

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    out_df = pd.DataFrame({
        'member_id': member_ids,
        'proba': proba,
        'pred': preds
    })
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f'Wrote predictions to {args.output}')


if __name__ == '__main__':
    main()
