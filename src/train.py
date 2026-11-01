
"""
Train and package the selected Bank Marketing model.

Selected model: Gradient Boosting
Preprocessing: SMOTENC + ColumnTransformer
Validation: Stratified 80:20 split
Final artifact: models/best_model.joblib
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ------------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_cleaned_data,
    split_features_target,
    split_data,
    build_smotenc_model_pipeline,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = REPORT_DIR / "evaluation_results.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
DECISION_THRESHOLD = 0.7419

BEST_PARAMETERS = {
    "model__learning_rate": 0.07027897634919636,
    "model__max_depth": 2,
    "model__min_samples_leaf": 2,
    "model__min_samples_split": 13,
    "model__n_estimators": 207,
    "model__subsample": 0.8,
}

# ------------------------------------------------------------
# 2. CREATE OUTPUT DIRECTORIES
# ------------------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("BANK MARKETING — MODEL TRAINING AND PACKAGING")
print("=" * 65)

# ------------------------------------------------------------
# 3. LOAD AND PREPARE DATA
# ------------------------------------------------------------

print("\n[1/7] Loading cleaned dataset...")

data = load_cleaned_data(DATA_PATH)

X, y = split_features_target(data)

print("Dataset shape:", data.shape)
print("Feature count:", X.shape[1])
print("Training target distribution:")
print(y.value_counts().sort_index())

# ------------------------------------------------------------
# 4. STRATIFIED TRAIN-VALIDATION SPLIT
# ------------------------------------------------------------

print("\n[2/7] Creating stratified train-validation split...")

X_train, X_valid, y_train, y_valid = split_data(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_valid))

# ------------------------------------------------------------
# 5. BUILD SELECTED GRADIENT BOOSTING PIPELINE
# ------------------------------------------------------------

print("\n[3/7] Building Gradient Boosting pipeline...")

validation_model = build_smotenc_model_pipeline(
    estimator=GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),
    random_state=RANDOM_STATE,
)

validation_model.set_params(**BEST_PARAMETERS)

print("Selected model: Gradient Boosting")
print("Hyperparameters:")
print(json.dumps(BEST_PARAMETERS, indent=4))

# ------------------------------------------------------------
# 6. TRAIN AND EVALUATE ON VALIDATION SPLIT
# ------------------------------------------------------------

print("\n[4/7] Training validation model...")

start_time = time.perf_counter()

validation_model.fit(X_train, y_train)

training_time = time.perf_counter() - start_time

probabilities = validation_model.predict_proba(X_valid)[:, 1]

predictions_default = (probabilities >= 0.50).astype(int)

predictions_selected = (
    probabilities >= DECISION_THRESHOLD
).astype(int)


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification and ranking metrics."""

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "customers_contacted": int(fp + tp),
    }


default_metrics = calculate_metrics(
    y_valid,
    predictions_default,
    probabilities,
)

selected_metrics = calculate_metrics(
    y_valid,
    predictions_selected,
    probabilities,
)

print("\nValidation metrics at threshold 0.50:")
print(json.dumps(default_metrics, indent=4))

print(
    f"\nValidation metrics at threshold "
    f"{DECISION_THRESHOLD}:"
)
print(json.dumps(selected_metrics, indent=4))

# ------------------------------------------------------------
# 7. RETRAIN ON COMPLETE DATA AND SAVE PIPELINE
# ------------------------------------------------------------

print("\n[5/7] Retraining final pipeline on complete dataset...")

final_model = build_smotenc_model_pipeline(
    estimator=GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),
    random_state=RANDOM_STATE,
)

final_model.set_params(**BEST_PARAMETERS)

final_model.fit(X, y)

print("\n[6/7] Saving model and metadata...")

joblib.dump(final_model, MODEL_PATH)

metadata = {
    "project": "Bank Marketing Subscription Prediction",
    "model_name": "Gradient Boosting",
    "model_type": "GradientBoostingClassifier",
    "random_state": RANDOM_STATE,
    "decision_threshold": DECISION_THRESHOLD,
    "threshold_selection": (
        "Selected using validation F1 analysis in Phase 7"
    ),
    "features": NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
    "numerical_features": NUMERICAL_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "excluded_features": ["duration"],
    "target_column": "y",
    "target_mapping": {"no": 0, "yes": 1},
    "preprocessing": [
        "SMOTENC",
        "StandardScaler",
        "OneHotEncoder",
    ],
    "hyperparameters": BEST_PARAMETERS,
    "training_dataset_rows": int(len(X)),
    "validation_rows": int(len(X_valid)),
    "validation_test_size": TEST_SIZE,
    "training_timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "model_artifact": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

# ------------------------------------------------------------
# 8. SAVE VALIDATION METRICS
# ------------------------------------------------------------

evaluation_results = {
    "project": "Bank Marketing Subscription Prediction",
    "model_name": "Gradient Boosting",
    "evaluation_type": "Stratified validation split",
    "random_state": RANDOM_STATE,
    "validation_size": int(len(X_valid)),
    "validation_positive_cases": int(y_valid.sum()),
    "validation_negative_cases": int((y_valid == 0).sum()),
    "training_time_seconds": float(training_time),
    "threshold_default": 0.50,
    "threshold_selected": DECISION_THRESHOLD,
    "metrics_at_default_threshold": default_metrics,
    "metrics_at_selected_threshold": selected_metrics,
    "note": (
        "Metrics are measured on the validation split. "
        "The final saved pipeline was subsequently retrained "
        "on the complete labelled dataset."
    ),
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(evaluation_results, file, indent=4)

# ------------------------------------------------------------
# 9. FINAL SUMMARY
# ------------------------------------------------------------

print("\n[7/7] TRAINING AND PACKAGING COMPLETED")

print("\nSaved artifacts:")
print("Model:", MODEL_PATH)
print("Metadata:", METADATA_PATH)
print("Evaluation:", METRICS_PATH)

print("\nFinal validation results at selected threshold:")
print(f"F1-score:  {selected_metrics['f1_score']:.4f}")
print(f"Precision: {selected_metrics['precision']:.4f}")
print(f"Recall:    {selected_metrics['recall']:.4f}")
print(f"ROC-AUC:   {selected_metrics['roc_auc']:.4f}")
print(f"PR-AUC:    {selected_metrics['pr_auc']:.4f}")

print("\n" + "=" * 65)
print("MODEL PACKAGING SUCCESSFUL")
print("=" * 65)
