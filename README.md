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



Required Deliverables
● A public Git repository containing a reproducible end-to-end solution.
● A README file detailing setup and run instructions, along with a concise description
of your approach.
● An executive presentation (3-5 slides) tailored for non-technical stakeholders.
● A CSV file containing a sorted list of the top 'n' members for outreach. This file must
include, at minimum, member_id, a prioritization score, and the member's rank.
Note: Use the provided test files (test_*.csv) to evaluate your final model and compare
your results to the test baseline metrics (*_baseline_test.txt).
Evaluation Criteria
Your submission will be evaluated based on the following aspects:
● Code Clarity and Readability
● Solution Robustness
● Visualization Quality
● Presenting Results
● Storytelling
Additional Guidance
To help you focus your efforts, please address the following in your approach and
documentation:
● Feature Selection: Explain which features you chose to use and why. Consider
domain relevance, data quality, and predictive power.
● Model Evaluation: Describe how you evaluate model performance and justify your
chosen metrics.
● Using Outreach Data in Modelling: The dataset includes an outreach event that
occurred between the observation period and the churn measurement window. You
are expected to incorporate this information into your modelling and explain how it
influences your approach and results.
● Selecting n (Outreach Size): Describe how you determine the optimal outreach
size. Is it driven only by cost, or are there other factors you considered?