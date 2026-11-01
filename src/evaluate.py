
"""
Evaluate a saved Bank Marketing model on a labelled dataset.

Usage:
    python src/evaluate.py --data data/raw/bank.csv

Important:
    Use an independent labelled dataset for an unbiased
    estimate of generalization. Do not treat training-data
    evaluation as unseen-data performance.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import split_features_target

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "external_evaluation.json"


def evaluate_model(data_path, output_path):
    """Evaluate the saved pipeline on labelled customer records."""

    data_path = Path(data_path)
    output_path = Path(output_path)

    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {data_path}"
        )

    print("=" * 65)
    print("BANK MARKETING — MODEL EVALUATION")
    print("=" * 65)

    print("\n[1/5] Loading model and metadata...")

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)

    threshold = float(metadata["decision_threshold"])

    print("Model:", metadata["model_name"])
    print("Threshold:", threshold)

    print("\n[2/5] Loading labelled evaluation data...")

    data = pd.read_csv(data_path, sep=None, engine="python")

    X, y = split_features_target(data)

    print("Dataset shape:", data.shape)
    print("Evaluation records:", len(data))

    print("\n[3/5] Generating predictions...")

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print("\n[4/5] Calculating evaluation metrics...")

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    results = {
        "model_name": metadata["model_name"],
        "evaluation_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_path": str(data_path),
        "evaluation_records": int(len(data)),
        "decision_threshold": threshold,
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "customers_contacted": int(fp + tp),
        "evaluation_note": (
            "Interpret as independent generalization performance "
            "only if the evaluation dataset was not used for "
            "model training, tuning, or threshold selection."
        ),
    }

    print("\nEvaluation results:")
    print(json.dumps(results, indent=4))

    print("\n[5/5] Saving evaluation results...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nSaved:", output_path)

    print("\n" + "=" * 65)
    print("MODEL EVALUATION COMPLETED")
    print("=" * 65)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the packaged Bank Marketing model."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to a labelled CSV evaluation dataset.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the evaluation JSON output.",
    )

    args = parser.parse_args()

    evaluate_model(args.data, args.output)
