WellCo Churn Prediction - Uplift Modeling Project
Project Overview
This repository contains a comprehensive uplift modeling solution for WellCo's member churn prediction and intervention optimization problem. The goal is to identify which members would benefit most from outreach interventions to reduce churn, using historical data that includes an outreach treatment experiment.

Problem Statement
WellCo wants to minimize member churn by targeting high-risk members with personalized outreach. The challenge is to:

Predict which members are likely to churn
Identify which members would respond positively to outreach (positive treatment effect)
Determine the optimal number of members to target (balancing cost vs. benefit)
Approach: Uplift Modeling (Causal Inference)
Unlike traditional churn prediction, this is a causal inference problem. We need to estimate the individual treatment effect (ITE) or conditional average treatment effect (CATE) for each member:

A positive uplift score means outreach reduces churn for that member.

Feature Selection
Domain-Relevant Features
Based on wellco_client_brief.txt and schema files, I focused on:

Clinical Features (High Priority)

ICD-10 codes: E11.9 (Type 2 Diabetes), I10 (Hypertension), Z71.3 (Dietary Counseling)
Claims frequency and cost patterns
Chronic condition indicators
Engagement Features

App usage frequency and recency
Web visit patterns
Session duration metrics
Feature-specific engagement (e.g., meal logging, activity tracking)
Temporal Features

Days since signup
Recent activity trends (last 7/30 days)
Engagement velocity (increasing vs. declining)
Feature Engineering Process
Feature Quality Considerations
Missing Data: Imputed with 0 for count-based features (assumes no engagement = 0 events)
Redundancy: Removed highly correlated features (|correlation| > 0.95) using feature_selection() in data_process.py
Treatment Interactions: Created interaction terms between key features and treatment status to capture heterogeneous effects
Using Outreach Data in Modeling
The dataset includes a historical outreach experiment where some members received outreach and others didn't. This is crucial for causal inference.

Why This Matters
Naive Approach (Wrong): Train a single model to predict churn, ignoring treatment

Problem: Doesn't tell us who benefits from outreach
Uplift Approach (Correct): Leverage treatment assignment to learn individual treatment effects

How I Incorporated Outreach Data
I implemented multiple meta-learners that explicitly model treatment effects:

1. T-Learner (Two-Model Approach)
2. S-Learner (Single Model with Treatment Feature)
Includes treatment as a feature
Learns treatment effect implicitly
Less flexible but more stable with smaller samples
3. DR-Learner (Doubly Robust)
Uses EconML library
Combines outcome modeling and propensity score weighting
Robust to model misspecification
4. X-Learner (Advanced Meta-Learner)
Imputes counterfactual outcomes
Trains models on imputed treatment effects
Weights by propensity scores
Propensity Score Integration
To account for non-random treatment assignment, I estimated propensity scores:

Propensity scores help:

Adjust for confounding (treated members may differ systematically)
Improve overlap between treatment groups
Stabilize variance in treatment effect estimates
Model Evaluation
Primary Metric: Qini AUC
For uplift modeling, I use Qini AUC (Area Under the Uplift Curve), which measures:

How well the model ranks members by treatment effect
Cumulative uplift gained by targeting top-ranked members
Why Qini AUC?

Business-Aligned: Directly measures incremental value from targeting
Causal: Accounts for treatment effects, not just churn probability
Rank-Based: Robust to calibration issues
Secondary Metrics
Overfitting Gap: Train Qini - Test Qini

Monitors generalization
Flag if gap > 0.01
Uplift Score Variance

Low variance indicates weak heterogeneous effects
High variance means strong personalization potential
Propensity Overlap

Visualize treated vs. control propensity distributions
Ensure common support for valid causal inference
Model Comparison
Model	Test Qini AUC	Overfitting Gap	Interpretation
Manual T-Learner	0.0124	0.0023	Good baseline
S-Learner (Elastic Net)	0.0118	0.0019	Stable, low variance
T-Learner (Tuned LightGBM)	0.0156	0.0031	Best performance
X-Learner	0.0142	0.0028	Good balance
DR-Learner	0.0139	0.0025	Robust
Winner: Tuned T-Learner with LightGBM (after hyperparameter optimization)

Selecting Optimal Outreach Size (n)
Methodology
I developed a Qini-based optimization approach to find the optimal number of members to target:

Factors Considered
Marginal Uplift Decay

Initial members have high treatment effects
Marginal benefit decreases as we go down the ranking
Optimal n is where marginal uplift = marginal cost
Cost-Benefit Analysis (if cost data available)

Outreach cost per member: C_outreach
Value of prevented churn: V_churn
Target where: Marginal_Uplift * V_churn >= C_outreach
Capacity Constraints

If WellCo has limited outreach capacity (e.g., max 2000 contacts/month)
Apply constraint: optimal_n = min(optimal_n_unconstrained, max_budget_n)
Risk Tolerance

Conservative: Target only high-confidence uplift scores
Aggressive: Expand to moderate uplift scores
Results for This Dataset
Interpretation:

Target the top 1,547 members ranked by uplift score
Expected to prevent ~12 additional churns compared to random targeting
Beyond this point, marginal uplift diminishes
Repository Structure
Key Findings
Treatment Heterogeneity: Strong evidence that outreach effects vary by member
Clinical Features: ICD-10 codes (diabetes, hypertension) are strong predictors of treatment effect
Engagement Patterns: Low recent engagement + high propensity score = high uplift potential
Optimal Targeting: Reach out to ~15-20% of at-risk members for maximum ROI
Next Steps
A/B Test Validation: Deploy model on a holdout population to validate uplift estimates
Cost Integration: Incorporate actual outreach costs for precise ROI calculation
Real-Time Scoring: Productionize model for ongoing member scoring
Feature Refinement: Add more granular engagement metrics (e.g., message response rates)
Dependencies
See requirements.txt for full list. Key libraries:

scikit-learn, lightgbm for modeling
econml for causal inference
sklift for uplift metrics and visualization
pandas, numpy for data manipulation