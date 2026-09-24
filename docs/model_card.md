# Model Card — Telco Customer Churn Prediction

## Model details
- **Type**: XGBoost binary classifier
- **Version**: 1.0
- **Author**: Oliver Chinn
- **Date**: September 2026
- **Trained on**: IBM Telco Customer Churn dataset (7,043 customers)

## What it does
Predicts the probability that a telecom customer will cancel their 
subscription (churn) within the next billing period, given their 
account and service usage features.

## Performance

| Metric | Value |
|---|---|
| ROC-AUC (test set) | ~0.845 |
| Precision (churn class) | ~0.65 |
| Recall (churn class) | ~0.54 |
| Training samples | 5,634 |
| Test samples | 1,409 |

### Comparison with baselines
| Model | ROC-AUC |
|---|---|
| XGBoost | ~0.845 |
| Random Forest | ~0.830 |
| Logistic Regression | ~0.845 |

## Key predictive features
1. Contract type (month-to-month = highest churn risk)
2. Tenure (new customers churn more)
3. Monthly charges
4. Number of services used
5. Internet service type

## Intended use
- Identifying at-risk customers for retention campaigns
- Prioritising customer success outreach
- Not intended for automated cancellation or penalty decisions

## Limitations and risks
- Trained on telecom data only — may not generalise to other 
  SaaS or subscription businesses without retraining
- Dataset is from one unnamed telecom company at one point in time —
  customer behaviour may have shifted since collection
- Class imbalance (26% churn) means the model is better at 
  identifying non-churners than churners — recall on the churn 
  class (~54%) means it misses roughly half of churning customers
- No demographic fairness analysis has been conducted — 
  gender and partner status are included as features which could
  introduce bias in certain deployment contexts

## Ethical considerations
This model should be used to trigger supportive interventions 
(outreach, offers) rather than punitive ones. Predictions carry 
uncertainty — a high churn probability score is a signal to 
investigate, not a definitive label.