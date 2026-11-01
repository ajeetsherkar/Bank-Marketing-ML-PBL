# Phase 9 — Model Packaging and Reproducibility

## 1. Objective
Package the selected machine learning model, preserve its metadata,
provide reusable training and evaluation scripts, and generate
predictions on unseen customer records.

## 2. Selected Model
- Model: Gradient Boosting Classifier
- Preprocessing: SMOTENC, StandardScaler and OneHotEncoder
- Input features: 15
- Excluded feature: duration
- Decision threshold: 0.7419

## 3. Model Artifacts
- models/best_model.joblib
- models/model_metadata.json

The trained pipeline and its configuration are saved for reuse.

## 4. Reproducibility Scripts
- src/train.py: trains and saves the model and metadata.
- src/evaluate.py: evaluates a saved model on labelled data.
- src/predict.py: generates subscription predictions for new customers.

## 5. Evaluation
Validation results from the training workflow at threshold 0.50:

- Accuracy: 0.7282
- Precision: 0.2205
- Recall: 0.5385
- F1-score: 0.3128
- ROC-AUC: 0.7171
- PR-AUC: 0.3211

At threshold 0.7419:

- Accuracy: 0.8807
- Precision: 0.4655
- Recall: 0.2596
- F1-score: 0.3333

The threshold was selected using validation data. It should be
validated on independent data before production use.

## 6. Prediction Results
Input file: data/raw/test.csv

- Input customers: 427
- Predictions generated: 427
- Predicted subscription: 0 yes, 427 no
- Very Low priority: 412
- Low priority: 15
- Missing prediction values: 0
- Probability range: 0.0126 to 0.3608

Output file: reports/predictions.csv

The test dataset has no ground-truth target labels, so predictive
accuracy cannot be calculated for these records.

## 7. Evaluation Limitation
The external evaluation script was tested using data/raw/bank.csv.
This is the same labelled dataset used to train the final model.
Therefore, its results are an execution/functionality check,
not an independent estimate of model generalization.

## 8. Final Audit
- Training script present.
- Evaluation script present.
- Prediction script present.
- Saved model loads successfully.
- Model metadata loads successfully.
- 15 expected features recorded.
- 427 predictions generated.
- All probabilities are within [0, 1].
- No missing prediction values.

## 9. Conclusion
Phase 9 successfully packages the selected Gradient Boosting
pipeline and provides reusable training, evaluation and prediction
scripts. Model artifacts and prediction outputs are saved for
reproducibility and further project development.
