WellCo Member Churn Reduction: Uplift Modeling Solution

This repository contains the end-to-end solution for the WellCo member churn reduction assignment. The primary goal is to identify and rank members who would be most positively influenced by a targeted outreach campaign (persuadables) to minimize overall churn.

The solution leverages Uplift Modeling (also known as Causal Inference) to estimate the causal effect of the outreach treatment on each member's churn probability, allowing for precise targeting.

1. Project Setup and Prerequisites

Prerequisites

To run this solution, you need to have Python 3.9+ installed. The project relies on several common data science and specialized uplift modeling libraries.

Environment Setup

Clone the repository:

git clone <your-repo-url>
cd wellco-churn-uplift


Install dependencies:
The core dependencies are managed within the main notebook's installation section. You will need pandas, numpy, scikit-learn, lightgbm, econml, and scikit-uplift (sklift).

A suggested method is to install the following packages:

pip install pandas numpy scikit-learn matplotlib seaborn lightgbm econml scikit-uplift


Note: econml may require specific configurations depending on your system.

Data Structure

Place all provided data files (web_visits.csv, app_usage.csv, claims.csv, churn_labels.csv, plus their corresponding test_ files and baseline files) into a data/ subdirectory within the project root.

.
├── data/
│   ├── web_visits.csv
│   ├── app_usage.csv
│   ├── claims.csv
│   ├── churn_labels.csv
│   ├── test_web_visits.csv
│   ├── test_app_usage.csv
│   ├── ... (other test/schema files)
├── notebooks/
│   └── main_notebook.ipynb
├── data_handler.py
├── data_process.py
├── trainers.py
└── README.md


2. How to Run the Solution

The entire analysis, including data loading, feature engineering, model training, evaluation, and final output generation, is contained within a single Jupyter Notebook.

Execution Steps

Launch Jupyter:

jupyter notebook notebooks/main_notebook.ipynb


Run the Notebook:
Execute all cells in the main_notebook.ipynb sequentially. The notebook is structured to perform the following steps:

Setup: Import necessary libraries and define paths.

Data Preparation: Load data and apply the feature engineering pipeline defined in data_handler.py.

Health Check & Split: Perform data quality checks and split the training set for validation.

Model Training: Train the chosen Uplift Model (Meta-Learner T-Learner with LightGBM base models) using the functions in trainers.py.

Evaluation: Visualize Qini curves and compare performance on validation data.

Final Prediction & Ranking: Predict uplift scores on the held-out test data.

Optimal 'n' Determination: Determine the optimal outreach size n based on the Qini curve analysis.

Output Generation: Save the final ranked member list to a CSV file in the outputs/ directory.

3. Solution Approach Overview

Problem Framing

This is an Uplift Modeling problem, not a standard classification problem. We are not just predicting who will churn (P(Churn)), but rather who will not churn if they receive outreach, compared to if they do not receive outreach.

The target metric is the Causal Treatment Effect (Uplift):

$$\text{Uplift}(X) = P(Y=1 | T=1, X) - P(Y=1 | T=0, X)
$$Where:

* $Y=1$: The member churns (positive outcome in this case, meaning we are modeling the probability of the negative event).
* $T=1$: The member received the outreach (Treatment).
* $T=0$: The member did not receive outreach (Control).
* $X$: The member's feature vector.

The goal is to identify members with a high **negative** uplift score (meaning outreach *decreases* the probability of churn), as these are the "persuadables." We prioritize members with the most negative uplift.

### Modeling Strategy

A **T-Learner (Two-Model Approach)** is employed, which is a powerful meta-learner for uplift modeling.

1.  **Model 1 (Treated):** Trained on treated group data ($T=1$) to predict $P(Y=1 | T=1, X)$.
2.  **Model 2 (Control):** Trained on control group data ($T=0$) to predict $P(Y=1 | T=0, X)$.
3.  **Uplift Calculation:** $\text{Uplift}(X) = \text{Model}_1(X) - \text{Model}_2(X)$.

**Base Estimators:** Both Model 1 and Model 2 utilize **LightGBM Classifiers** due to their efficiency, handling of categorical features, and strong performance in predictive modeling tasks.

### Feature Engineering (`data_handler.py`)

Feature engineering focuses on extracting aggregated usage statistics and capturing behavioral changes relative to the outreach event:

| Data Source | Features Created | Domain Relevance & Rationale |
| :--- | :--- | :--- |
| **`churn_labels.csv`** | **`treatment`** (1/0), `age_group`, `gender` | Baseline demographics. The **`treatment`** column is crucial for uplift modeling. |
| **`app_usage.csv`** | Total app events, mean session length, **ratio of pre-outreach to post-outreach activity**. | Captures overall engagement and behavioral shift (potential influence of outreach). |
| **`web_visits.csv`** | Total web events, page views per visit, **event counts before and after the `MID_DATE`**. | Measures digital engagement and whether activity changed after the observation period midpoint. |
| **`claims.csv`** | Total claims count, count of priority ICD codes, days since last claim. | Clinical necessity and service utilization. Frequency and severity of health issues. |

### Model Evaluation and Metrics

The primary evaluation metric for uplift models is the **Area Under the Qini Curve (AUQC)**, also known as the Qini coefficient.

* **Qini Curve:** Measures the cumulative uplift gained by targeting members ranked by the uplift score, compared to a random targeting strategy.
* **Optimal `n` Determination:** The optimal outreach size `n` is selected by finding the point on the Qini curve where the **cumulative uplift is maximized**. This ensures the maximum positive impact is achieved, balancing cost (targeting more people) against reward (maximum churn reduction). The peak of the Qini curve indicates the optimal number of members to target.

### Incorporating Outreach Data

The outreach event occurred between the 14-day observation window and the churn measurement window. The **`churn_labels.csv`** file is the source of the `treatment` (outreach) variable.

This approach incorporates the outreach data as follows:

1.  **Training:** The `treatment` column is used to split the training data into two distinct groups (Treated and Control) to train the two separate base models of the T-Learner.
2.  **Feature Engineering:** Features were engineered to reflect activity **before** the outreach event (July 1 - July 14) and activity **before and after** the midpoint date (July 7) to assess underlying behavior *before* any potential influence.

-----

## 4\. Deliverables Generated

Upon successful execution of the notebook, the following critical deliverable will be saved in the `outputs/` folder:

* `outreach_list_[timestamp].csv`: A CSV file containing the ranked list of the top 'n' members for prioritized outreach.
* **Columns:** `member_id`, `prioritization_score` (the calculated negative uplift score), and `rank`.$$