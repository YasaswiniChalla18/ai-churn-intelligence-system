# AI-Powered Customer Churn Intelligence System

## Business Problem
Telecom companies know *which* customers are leaving, but often don't understand
*why* — which makes it hard to act. This project predicts churn risk for each
customer, explains the top factors behind each prediction, and turns that into
a retention recommendation a business team could actually act on.

## Dataset
IBM Telco Customer Churn dataset — 7,043 customers, 21 features
(demographics, account info, services subscribed, billing).
Source: IBM sample dataset, publicly available on GitHub and Kaggle
(`blastchar/telco-customer-churn`).

## Pipeline
1. **`src/01_eda.py`** — Cleans data (fixes `TotalCharges` type, handles blanks
   for brand-new customers), computes churn rate, and visualizes churn by
   contract type, tenure, and monthly charges.
2. **`src/02_model.py`** — One-hot encodes categorical features, splits data
   (stratified, since churn is imbalanced at ~26%), trains Logistic Regression
   as a baseline and Random Forest as the main model, evaluates with
   precision/recall/ROC-AUC (not just accuracy, since accuracy is misleading
   on imbalanced data).
3. **`src/03_shap_explain.py`** — Uses SHAP TreeExplainer to show (a) which
   features drive churn globally across all customers, and (b) why one specific
   high-risk customer is predicted to churn.

## Key Results (from this run — reproducible by re-running the scripts)
- Overall churn rate: **26.5%**
- Churn by contract type: Month-to-month **42.7%**, One-year **11.3%**, Two-year **2.8%**
- Random Forest: ROC-AUC **0.844**, recall on churners **0.78**
  (recall is prioritized over raw accuracy — missing an actual churner is
  costlier to the business than a false alarm)
- Top global churn drivers (via SHAP): contract type, tenure, internet service
  type, monthly charges, payment method

## Example Output (business-facing framing)
> Customer [row 1109]: 92% predicted churn probability.
> Main drivers: short tenure, fiber optic internet service, high total
> charges relative to tenure, and electronic check payment method.
> Recommendation: proactive retention outreach — offer a loyalty discount or
> contract upgrade incentive before the next billing cycle.

## Tech Stack
- Python, pandas, scikit-learn
- SHAP (model explainability)
- matplotlib/seaborn (EDA + result visualization)
- (Optional next step: Power BI/Tableau dashboard on top of `outputs/cleaned_data.csv`
  and model predictions, for a business-facing view)

## How to Run
```bash
pip install -r requirements.txt
python src/01_eda.py
python src/02_model.py
python src/03_shap_explain.py
```
All plots and intermediate data land in `outputs/`.

## Honest Limitations
- This is a single static snapshot dataset — no ability to track how a customer's
  service choices evolve over time (a limitation of the source data, not the model).
- The dataset is a well-known public sample, not live company data — a real
  deployment would need pipeline integration with live billing/usage systems.
- Class imbalance was handled with `class_weight="balanced"`; SMOTE or threshold
  tuning could be explored further to push recall higher if false positives are
  cheap for the business in question.

## Why This Project Stands Out
It doesn't stop at "the model predicts churn." It explains *why*, at both a
global level (which features matter most overall) and an individual level
(why this one customer is at risk) — which is the difference between a model
and a decision-support tool a business can actually use.
