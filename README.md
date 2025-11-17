# 🚀 WellCo Churn Prediction - Uplift Modeling Project

A comprehensive causal inference solution to optimize member outreach for churn reduction by estimating Individual Treatment Effects (ITE).

---

## ⚡ Quick Start - How to Run

Follow these steps to set up the environment and execute the project notebooks.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/eladlaks/wellco_churn.git
    ```

2.  **Create a virtual environment (Python 3.10 required):**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the environment:**
    * **macOS/Linux:** `source .venv/bin/activate`
    * **Windows (Command Prompt):** `.venv\Scripts\activate`

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the main notebook** (`main_analysis.ipynb`) to replicate the feature engineering, modeling, and optimization steps.

---

## 💡 Project Overview: Optimizing Intervention with Uplift Modeling

This project moves beyond standard churn prediction to address the **causal inference** problem of **intervention optimization**.

The goal is to answer: **"For which members will outreach successfully prevent churn?"**

### Problem Statement

WellCo seeks to minimize member churn by targeting high-risk individuals with personalized outreach. The core challenges are:
* Predicting which members are **likely to churn** (Outcome Model).
* Identifying which members will **respond positively to outreach** (Treatment Effect).
* Determining the **optimal number of members to target** (Cost-Benefit Balance).

### Approach: Uplift Modeling (Causal Inference)

We estimate the **Individual Treatment Effect (ITE)** or **Conditional Average Treatment Effect (CATE)** for each member.

$$
\text{Uplift Score} = \text{P}(\text{Churn}=0 \mid \text{Outreach}=1) - \text{P}(\text{Churn}=0 \mid \text{Outreach}=0)
$$

A **positive uplift score** means outreach increases the probability of *non-churn* (i.e., it successfully prevents churn) for that specific member.

---

## 🚧 Modeling Challenges & Assumptions

Causal inference on observational data presents inherent challenges that limit performance compared to standard predictive tasks.

### Low Treatment Effect Signal

Initial analysis showed that the **Average Treatment Effect (ATE)** across the entire population is **very small** (close to zero). This implies:
* The intervention (outreach) has a **low overall impact** on preventing churn for the average member.
* The low signal makes it **significantly harder** for the uplift models to find and differentiate the few members with a strong **heterogeneous treatment effect (Uplift)**.
* Consequently, the overall **Qini AUC scores are low** ($< 0.05$), reflecting the weak underlying causal signal we are trying to isolate.

### Non-Randomized Treatment Assignment (Confounding)

The original outreach data was **not a perfectly randomized controlled trial (RCT)**, meaning treatment assignment may have been influenced by observed member characteristics. This introduces **selection bias (confounding)**.

* **Impact:** If members *systematically* more likely to churn were targeted (or vice-versa), the observed difference between treated and control groups is not purely the treatment effect.
* **Mitigation Attempt:** I addressed this by estimating **Propensity Scores** to adjust for the non-random assignment. This ensures better covariate balance and common support, making the causal estimates more robust, though it cannot fully eliminate confounding from unobserved variables.

---
## 🔬 Feature Selection & Engineering ⚙️

Based on `wellco_client_brief.txt` and schema analysis, features were meticulously categorized and engineered to maximize predictive power and **treatment interaction capture**, which is crucial for estimating **heterogeneous uplift effects**.

---

### Domain-Relevant Features

The analysis focused on creating features that capture a member's **clinical risk**, **digital engagement intensity**, and their **temporal behavior patterns**.

| Category | Key Indicators | Specific Examples (Features) |
| :--- | :--- | :--- |
| **Clinical** | **Claims frequency/cost**, Chronic condition flags, **ICD-10 codes** related to high-risk chronic diseases. | `total_claims`, `unique_icd_codes`, `has_icd_I10` (Hypertension), `has_icd_E11_9` (Type 2 Diabetes), `has_icd_Z71_3` (Dietary Counseling). |
| **Engagement** | **App/Web usage frequency**, recency, diversity, and feature-specific tracking. | `app_sessions`, `active_days`, `total_web_visits`, `unique_domains`, `wellco_domain_visits`, `ratio_wellco_domain`. |
| **Temporal** | **Member lifetime**, activity patterns, and **velocity** (recent vs. past activity). | `tenure_days`, `early_usage`, `late_usage`, `usage_trend`, `visit_trend`, `early_claims`, `late_claims`, `claim_trend`. |

---

### Feature Engineering Process

The process included specific transformations and selection steps to ensure data quality and relevance for causal modeling:

* **Missing Data Imputation:**
    * Count-based features, such as `app_sessions` or `total_claims`, were **imputed with 0** (assuming the absence of a record signifies zero events or engagement).

* **Behavioral Trends & Velocity:**
    * **Trend features** (`usage_trend`, `visit_trend`, `claim_trend`) were calculated as the difference or ratio between **recent** (`late_*`) and **past** (`early_*`) activity periods. This is vital as **declining engagement is a strong churn signal**.
    * **Tenure** (`tenure_days`) was used as a normalization factor for many rate-based features.

* **Redundancy Reduction:**
    * Highly correlated features ($|\text{correlation}| > 0.95$) were removed using `feature_selection()` in `data_process.py` to prevent multicollinearity, which is particularly important for meta-learner stability.

* **Treatment Interactions (Heterogeneity Capture):**
    * **Explicit interaction terms** (e.g., Feature $\times$ Treatment Status) were created between key features (like `has_icd_E11_9`, `app_sessions`) and the treatment status. This allows the meta-learners to explicitly model **heterogeneous effects**, capturing *how* the outreach impact differs based on a member's clinical or engagement profile.
---

## 📊 Modeling & Causal Inference

The historical outreach experiment data is leveraged to train models that explicitly estimate the individual treatment effect.

### Why Uplift Matters

| Approach | Goal | Problem |
| :--- | :--- | :--- |
| **Naive (Wrong)** | Predict likelihood of churn ($\text{P}(\text{Churn})$) | Ignores treatment effect; doesn't identify who **benefits** from outreach. |
| **Uplift (Correct)** | Estimate individual treatment effect ($\text{ITE}$) | Leverages treatment assignment to learn heterogeneous impact. |

### Meta-Learners Implemented

| Learner | Core Methodology | Key Library |
| :--- | :--- | :--- |
| **T-Learner** | Two separate models (Treated vs. Control) to estimate CATE. | `sklift` / Manual |
| **S-Learner** | Single model including `treatment` as a feature. Learns effect implicitly. | `sklift` / Manual |
| **X-Learner** | Imputes counterfactual outcomes, trains on imputed treatment effects, weighted by propensity scores. | `sklift` |
| **DR-Learner** | Doubly Robust; combines outcome modeling and propensity score weighting. | `EconML` |

### Propensity Score Integration

Propensity scores ($\text{P}(\text{Treatment} \mid \text{Features})$) were estimated to:
* **Adjust for Confounding:** Account for systematic differences between treated and control groups.
* **Improve Overlap:** Ensure sufficient common support for valid causal inference.

---

## 📈 Model Evaluation & Comparison

### Primary Metric: Qini AUC

The main metric for uplift modeling is the **Qini AUC (Area Under the Uplift Curve)**.

* **Business-Aligned:** Measures the cumulative incremental value (prevented churn) gained by targeting members ranked by their uplift score.
* **Causal:** Directly evaluates the model's ability to rank by treatment effect.



**Winner: Tuned T-Learner with LightGBM** (Selected after hyperparameter optimization).

---

## 🎯 Selecting Optimal Outreach Size ($n$)

The optimal number of members to target is determined using a Qini-based optimization approach, balancing marginal uplift decay against cost.

### Methodology

The optimal $n$ is determined where the **Marginal Uplift** begins to equal the **Marginal Cost**.

* **Factors Considered:** Marginal Uplift Decay, Capacity Constraints (max budget), and Risk Tolerance.
* **Cost-Benefit Integration (Future):** Target where $\text{Marginal\_Uplift} \times \text{V}_{\text{churn}} \ge \text{C}_{\text{outreach}}$.

### Results for This Dataset

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Optimal Targeting Size ($\mathbf{n}$)** | **1,547** members | Target $\approx 15-20\%$ of the at-risk population. |
| **Expected Additional Preventions** | $\sim 12$ churns | Compared to a random targeting strategy. |

---

## 🔮 Future Work & Continuous Improvement

To productionize this solution and enhance its predictive performance and robustness, the following steps are recommended:

* **Solution for Unrandomized Treatment:** Explore more advanced causal methods.
* **Comprehensive Hyperparameter Tuning:** Implement automated, grid-based hyperparameter optimization (e.g., using `Optuna` or `scikit-optimize`) for **all** base models used within the T-Learner, S-Learner, and X-Learner to maximize CATE estimation.
* **MLOps - Experiment Logging:** Integrate the training pipeline with **Weights & Biases (WandB)** or a similar platform to log and track:
    * Model architectures and hyperparameters.
    * Test and Train Qini AUC for every experiment run.
    * Propensity score distributions and overlap diagnostics.
* **A/B Test Validation:** Deploy the model on a holdout population to validate uplift estimates.
* **Cost Integration:** Incorporate actual outreach costs for precise ROI calculation.
* **Feature Refinement:** Add more granular engagement metrics.

---
