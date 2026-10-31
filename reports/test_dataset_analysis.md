# Test Dataset Analysis

## 1. Dataset Overview

- File: data/raw/test.csv
- Records: 427
- Columns: 17
- Target variable: y
- Target labels available: No

## 2. Schema Comparison

The training dataset contains 16 predictor features and the
target variable y.

The test dataset contains the same 16 predictor features
and an additional Id column.

| Dataset | Records | Columns |
|---|---:|---:|
| Training | 4,521 | 17 |
| Test | 427 | 17 |

### Schema Differences

- Target column y is absent from the test dataset.
- Id is present in the test dataset but absent from training.
- All 16 predictor features are shared.

## 3. Data Quality Results

| Check | Result |
|---|---:|
| Missing values | 0 |
| Duplicate records | 0 |
| Duplicate IDs | 0 |
| Minimum ID | 1 |
| Maximum ID | 427 |
| Sample submission records | 427 |
| Sample submission IDs match test IDs | Yes |

## 4. Sample Submission Analysis

The supplied sample submission contains:

- Id column
- y column
- 427 records
- All initial y values equal to 0.5

These values are treated as placeholders, not actual model
predictions.

The exact submission format and whether y requires
probabilities or class labels must be confirmed from the
competition instructions.

## 5. Intended Usage

The test dataset is reserved for generating predictions
after model development and evaluation.

Because actual target labels are unavailable, it cannot
be used for supervised model evaluation.

The Id column will be retained for submission mapping
but excluded from model predictors.

## 6. Important Modelling Consideration

The duration feature represents the duration of the last
marketing contact. It may only be known after the call ends.

Therefore, duration may cause information leakage if the
intended prediction must be made before contacting a customer.

Its inclusion or exclusion will be formally decided during
feature engineering and model development.

---

Project: Bank Marketing Subscription Prediction
Phase: 2 - Dataset Collection and Understanding
Author: Ajeet Sherkar
