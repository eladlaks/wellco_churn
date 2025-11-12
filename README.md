# wellco_churn

This repository contains data artifacts for a churn prediction exercise and a small training scaffold (notebook + script) that trains an XGBoost model and compares it to the baseline metrics included in `data/`.

Files added by the recent change:

- `notebooks/main.ipynb` — main interactive notebook that loads data, aggregates simple features, trains an XGBoost model, evaluates metrics, and compares ROC-AUC to the baseline found in `data/auc_baseline_test.txt`.
- `src/train.py` — small command-line script that performs the same steps as the notebook for quick runs.
- `requirements.txt` — minimal dependencies to run the notebook/script.

Quick start (Windows, cmd.exe):

1. Create a virtual environment and install dependencies:

	python -m venv .venv
	.venv\Scripts\activate
	pip install -r requirements.txt

2. Run the training script:

	python src\train.py

Or open `notebooks/main.ipynb` in Jupyter and run the cells interactively.

Notes:
- The notebook and script expect the CSVs under `data/train/` and `data/test/`: `app_usage.csv`, `web_visits.csv`, `claims.csv`, `churn_labels.csv`.
- Baseline AUC is read from `data/auc_baseline_test.txt` when present and printed for comparison.

If you want, I can add a small test or CI step that runs `src/train.py` on a tiny synthetic sample and asserts it completes.