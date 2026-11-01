# Phase 8 — Business Insights and Recommendations

## 1. Objective

Translate the machine learning findings into practical recommendations for improving bank marketing campaigns and identifying potential term-deposit subscribers.

## 2. Dataset Overview

- Total customer records: 4,521
- Non-subscribers: 4,000 (88.48%)
- Subscribers: 521 (11.52%)
- Missing values: 0
- Duplicate records: 0

The dataset is imbalanced, so marketing decisions should not rely on accuracy alone.

## 3. Customer Segment Analysis

### Subscription by Job

| Job | Customers | Subscribers | Subscription Rate |
|---|---:|---:|---:|
| Retired | 230 | 54 | 23.48% |
| Student | 84 | 19 | 22.62% |
| Management | 969 | 131 | 13.52% |
| Blue-collar | 946 | 69 | 7.29% |

Retired and student customers show relatively high subscription rates. These segments may deserve further targeted campaign analysis.

### Subscription by Marital Status

| Marital Status | Customers | Subscribers | Subscription Rate |
|---|---:|---:|---:|
| Divorced | 528 | 77 | 14.58% |
| Single | 1,196 | 167 | 13.96% |
| Married | 2,797 | 277 | 9.90% |

### Subscription by Education

| Education | Customers | Subscribers | Subscription Rate |
|---|---:|---:|---:|
| Tertiary | 1,350 | 193 | 14.30% |
| Secondary | 2,306 | 245 | 10.62% |
| Unknown | 187 | 19 | 10.16% |
| Primary | 678 | 64 | 9.44% |

### Subscription by Contact Type

| Contact Type | Customers | Subscribers | Subscription Rate |
|---|---:|---:|---:|
| Telephone | 301 | 44 | 14.62% |
| Cellular | 2,896 | 416 | 14.36% |
| Unknown | 1,324 | 61 | 4.61% |

### Previous Campaign Outcome

| Outcome | Customers | Subscribers | Subscription Rate |
|---|---:|---:|---:|
| Success | 129 | 83 | 64.34% |
| Other | 197 | 38 | 19.29% |
| Failure | 490 | 63 | 12.86% |
| Unknown | 3,705 | 337 | 9.10% |

Previous campaign success is strongly associated with subscription in this dataset.

## 4. Selected Model

Gradient Boosting was selected based on comparative validation performance.

| Metric | Tuned Validation Score |
|---|---:|
| F1-score | 0.3128 |
| PR-AUC | 0.3211 |
| ROC-AUC | 0.7171 |

These results indicate useful predictive ranking, although further improvement and independent testing are required before production use.

## 5. Feature Importance

The top features identified by the selected Gradient Boosting model include:

| Feature | Importance |
|---|---:|
| Contact: Cellular | 0.29163 |
| Previous contact days | 0.08893 |
| Account balance | 0.06968 |
| Personal loan: Yes | 0.06427 |
| Last contact day | 0.05086 |
| Campaign contacts | 0.04894 |

Feature importance indicates predictive contribution, not causation.

## 6. Predicted Probability Analysis

The validation dataset contains 905 customers.

| Probability Group | Customers | Actual Subscribers | Actual Subscription Rate |
|---|---:|---:|---:|
| 0–20% | 372 | 17 | 4.57% |
| 20–40% | 162 | 16 | 9.88% |
| 40–60% | 210 | 29 | 13.81% |
| 60–80% | 126 | 22 | 17.46% |
| 80–100% | 35 | 20 | 57.14% |

The highest probability group has the highest observed subscription rate. This supports using predicted probabilities to prioritize customers, while recognizing that predictions are not guarantees.

## 7. Marketing Recommendations

1. Prioritize customers with predicted subscription probabilities above 60% for personal calls and follow-ups.
2. Use targeted offers and follow-up communication for medium-probability customers.
3. Consider lower-cost communication channels for low-probability customers.
4. Use previous campaign outcomes as one input when prioritizing customers.
5. Allocate calling staff and campaign budgets according to predicted customer interest and available resources.
6. Monitor actual campaign outcomes and periodically evaluate model performance.

## 8. Model Limitations

- The dataset contains only 4,521 records.
- The target variable is imbalanced, with 11.52% positive cases.
- Predictive relationships do not establish causal relationships.
- Customer behavior and campaign effectiveness may change over time.
- Predicted probabilities do not guarantee subscription outcomes.
- Validation results are not an independent estimate of future performance.
- Independent testing on future campaign data is required before production deployment.
- Feature importance can vary across models.

## 9. Ethical Considerations

- Protect customer privacy and securely handle personal information.
- Check for unfair discrimination across customer groups.
- Maintain transparency about how predictions influence marketing prioritization.
- Avoid excessive or unwanted marketing calls.
- Retain human oversight in final marketing decisions.
- Use customer data only for authorized business purposes and follow applicable data-protection requirements.

## 10. Final Business Conclusion

The Gradient Boosting model provides a useful way to rank potential subscribers and understand customer characteristics associated with subscription.

Combining customer segmentation, feature importance and predicted probabilities can help the bank make more informed marketing decisions and allocate campaign resources more effectively.

**Final recommendation:** Use the selected model as a decision-support tool for targeted marketing, subject to independent testing, fairness checks and continuous performance monitoring.
