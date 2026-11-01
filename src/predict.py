
"""
Predict Bank Marketing subscriptions for new customers.

Usage:
    python src/predict.py --input data/raw/test.csv
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "predictions.csv"


def predict_customers(input_path, output_path):
    """Generate subscription predictions for unseen customer records."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("=" * 65)
    print("BANK MARKETING — CUSTOMER PREDICTION")
    print("=" * 65)

    # Load model and metadata
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)

    threshold = float(metadata["decision_threshold"])
    required_features = metadata["features"]

    # Load customer data
    customers = pd.read_csv(input_path, sep=None, engine="python")

    # Remove target if supplied accidentally
    X = customers.drop(
        columns=["y", "duration", "subscribed"],
        errors="ignore",
    )

    missing = set(required_features) - set(X.columns)
    extra = set(X.columns) - set(required_features)

    if missing:
        raise ValueError(
            f"Missing required features: {sorted(missing)}"
        )

    if extra:
        print("Ignoring extra columns:", sorted(extra))

    # Enforce the exact feature order used during training
    X = X[required_features]

    print("\nInput customers:", len(X))
    print("Model:", metadata["model_name"])
    print("Decision threshold:", threshold)

    # Generate predictions
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # Create output
    results = customers.copy()

    results["subscription_probability"] = probabilities
    results["predicted_class"] = predictions
    results["predicted_subscription"] = pd.Series(
        predictions,
        index=results.index,
    ).map({0: "no", 1: "yes"})

    results["marketing_priority"] = pd.cut(
        probabilities,
        bins=[-0.000001, 0.20, 0.40, 0.60, 0.80, 1.000001],
        labels=[
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High",
        ],
    )

    # Save predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\nPrediction summary:")
    print(results["predicted_subscription"].value_counts())

    print("\nMarketing priority:")
    print(results["marketing_priority"].value_counts())

    print("\nFirst five predictions:")
    print(
        results[
            [
                "subscription_probability",
                "predicted_subscription",
                "marketing_priority",
            ]
        ].head().to_string(index=False)
    )

    print("\nPredictions saved:", output_path)

    print("\n" + "=" * 65)
    print("CUSTOMER PREDICTION COMPLETED")
    print("=" * 65)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Bank Marketing subscriptions."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV containing customer features.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path.",
    )

    args = parser.parse_args()

    predict_customers(args.input, args.output)
