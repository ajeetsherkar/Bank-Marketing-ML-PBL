# Phase 5: Feature Engineering and Feature Selection

## Project
Bank Marketing Subscription Prediction

## 1. Objective

Investigate engineered features and apply feature-selection techniques to identify useful predictors of bank marketing subscription.

## 2. Dataset

- Observations: 4,521
- Original columns: 17
- Target: y
- Training records: 3,616
- Validation records: 905
- Training-validation split: Stratified 80:20
- Random state: 42
- Duration excluded from predictive features due to potential target leakage.

## 3. Engineered Features

Five candidate features were investigated:

1. age_group
2. balance_group
3. has_previous_contact
4. total_contacts
5. campaign_intensity

Balance thresholds were learned from training data only.

Original features were retained for comparison.

## 4. Feature Selection Techniques

The following techniques were investigated:

- Pearson correlation analysis
- Mutual Information
- Random Forest impurity-based importance
- Permutation importance using validation Average Precision
- Feature redundancy analysis
- Controlled feature-set comparison

## 5. Correlation Findings

Two highly correlated feature pairs were identified:

| Feature pair | Correlation |
|---|---:|
| campaign and total_contacts | 0.8771 |
| pdays and has_previous_contact | 0.8638 |

These relationships indicate overlapping information. No automatic feature removal was performed.

## 6. Mutual Information Findings

The highest-ranked features included:

| Feature | MI score |
|---|---:|
| pdays | 0.0634 |
| poutcome | 0.0262 |
| month | 0.0221 |
| previous | 0.0144 |
| age | 0.0113 |

MI scores were used as a screening method, not as the sole basis for feature removal.

## 7. Random Forest Importance

The leading impurity-based features included:

| Feature | Importance |
|---|---:|
| balance | 0.1059 |
| age | 0.0917 |
| day | 0.0887 |
| total_contacts | 0.0402 |
| campaign | 0.0363 |

Impurity-based importance was interpreted with caution.

## 8. Permutation Importance

Permutation importance used validation Average Precision (PR-AUC).

Leading features:

| Feature | Mean importance |
|---|---:|
| poutcome_success | 0.069804 |
| poutcome_unknown | 0.011802 |
| pdays | 0.010793 |
| month_may | 0.010385 |
| previous | 0.008307 |

## 9. Feature-Set Comparison

Four Random Forest experiments were compared.

| Experiment | PR-AUC | Recall | F1-score |
|---|---:|---:|---:|
| A: Full features | 0.3387 | 0.4327 | 0.3571 |
| B: Without total_contacts | 0.3432 | 0.4231 | 0.3492 |
| C: Without has_previous_contact | 0.3385 | 0.4519 | 0.3643 |
| D: Without both | 0.3397 | 0.5000 | 0.3969 |

Experiment B achieved the highest PR-AUC.
Experiment D achieved the highest recall and F1-score.

Differences were relatively small.

## 10. Feature Selection Decision

No final feature set was approved solely from these experiments.

The four feature configurations remain candidates for further evaluation.

The final feature set should be selected after additional model validation and comparison.

## 11. Leakage Prevention

- Duration was excluded from predictive features.
- Target y was never included in model inputs.
- Balance thresholds were derived from training data.
- Encoders and model pipelines were fitted using training data only.
- Validation data was kept separate during feature selection fitting.

## 12. Visualizations

Saved under `images/feature_engineering/`:

1. 01_feature_correlation.png
2. 02_mutual_information.png
3. 03_random_forest_importance.png
4. 04_permutation_importance.png

## 13. Conclusion

Phase 5 investigated five candidate engineered features and multiple feature-selection techniques.

The experiments identified potentially redundant features and demonstrated different trade-offs between PR-AUC, recall and F1-score.

Final feature selection will be confirmed through further model development and validation.

## 14. Business Insights

- Previous campaign outcomes (`poutcome`) and contact history (`pdays`, `previous`) provide useful predictive information.
- Customer age and account balance contribute to identifying differences in subscription behaviour.
- Campaign-related features contain overlapping information, so redundancy should be considered during final feature selection.
- The best feature set depends on the evaluation objective: PR-AUC, recall, and F1-score show different trade-offs.
- The final model should prioritize reliable identification of potential subscribers while considering the cost of false positives and false negatives.
