# Phase 7 — Model Evaluation and Validation Report

## 1. Project Overview

**Project:** Bank Marketing Subscription Prediction  
**Phase:** 7 — Model Evaluation and Validation  
**Selected Model:** Gradient Boosting  
**Objective:** Evaluate model performance, investigate generalization, optimize the classification threshold and assess business impact.

## 2. Dataset Overview

- Total observations: 4,521
- Total features: 16 (including target)
- Negative class (no): 4,000 (88.48%)
- Positive class (yes): 521 (11.52%)
- Missing values: 0
- Duplicate rows: 0

The dataset is imbalanced, making precision, recall, F1-score and PR-AUC particularly important.

## 3. Models Evaluated

Five classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. Gradient Boosting

All models were evaluated using a consistent stratified training-validation split and the same evaluation metrics.

## 4. Selected Model Performance

Gradient Boosting was selected based on its overall tuned validation performance.

| Metric | Validation Score |
|---|---:|
| Accuracy | 0.7282 |
| Precision | 0.2205 |
| Recall | 0.5385 |
| F1-score | 0.3128 |
| ROC-AUC | 0.7171 |
| PR-AUC | 0.3211 |

### Confusion Matrix

At the default classification threshold of 0.50:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 603 | 198 |
| Actual Yes | 48 | 56 |

The model identified 56 actual subscribers and missed 48 subscribers.

## 5. Overfitting and Generalization

Training and validation performance were compared.

| Metric | Training | Validation |
|---|---:|---:|
| Accuracy | 0.7494 | 0.7282 |
| Precision | 0.2548 | 0.2205 |
| Recall | 0.6091 | 0.5385 |
| F1-score | 0.3593 | 0.3128 |
| ROC-AUC | 0.7652 | 0.7171 |
| PR-AUC | 0.3949 | 0.3211 |

The relatively small training-validation gaps suggest that the selected model does not show severe overfitting.

## 6. Classification Threshold Optimization

The default classification threshold of 0.50 was compared with alternative thresholds.

- Default threshold: 0.50
- Selected threshold: 0.7419
- Reproducible Phase 9 validation F1-score at threshold 0.7419: 0.3333
- Reproducible Phase 9 precision at threshold 0.7419: 0.4655
- Reproducible Phase 9 recall at threshold 0.7419: 0.2596

The original Phase 7 notebook recorded a maximum validation F1-score of 0.3436 at threshold 0.7419. The later Phase 9 training workflow produced slightly different validation probabilities and metrics at the same threshold. The Phase 9 results are used as the final reproducible evaluation results.

This demonstrates the trade-off between reducing unnecessary marketing calls and identifying more potential subscribers.

**Important:** The threshold was selected using validation data. Its performance must be confirmed on independent unseen data before deployment.

## 7. Cross-Validation Stability

Stratified five-fold cross-validation was performed on the training data.

| Metric | CV Mean | CV Standard Deviation |
|---|---:|---:|
| Accuracy | 0.7411 | 0.0157 |
| Precision | 0.2233 | 0.0085 |
| Recall | 0.5039 | 0.0715 |
| F1-score | 0.3085 | 0.0192 |
| ROC-AUC | 0.7103 | 0.0292 |
| PR-AUC | 0.3202 | 0.0352 |

The cross-validation results indicate reasonably consistent F1 performance across folds.

## 8. Business Impact

The model can help a bank prioritize customers who are more likely to subscribe to a term deposit.

- Higher precision can reduce unnecessary marketing calls.
- Higher recall can help identify more potential subscribers.
- Threshold selection should reflect campaign costs and the value of successful subscriptions.

The final operating threshold should be selected according to the bank's business requirements.

## 9. Evaluation Limitations

- The validation dataset was used for model comparison and threshold selection.
- Validation performance is not an independent estimate of real-world performance.
- No separate labeled test dataset was used for final performance confirmation in this phase.
- Business profitability cannot be calculated without campaign costs and subscription revenue information.

## 10. Final Conclusion

Gradient Boosting was selected based on the comparative model evaluation.

In the final reproducible Phase 9 validation workflow, the model achieved an F1-score of 0.3128 at threshold 0.50 and 0.3333 at threshold 0.7419. ROC-AUC was 0.7171 and PR-AUC was 0.3211.

Cross-validation demonstrated reasonably stable performance. Threshold analysis also showed that the classification threshold can be adjusted to balance precision and recall according to business requirements.

**Phase 7 deliverables:** Model evaluation notebook, evaluation report, model comparison results, validation metrics and evaluation visualizations.
