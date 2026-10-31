"""
Reusable preprocessing utilities for the Bank Marketing ML project.

Responsibilities:
- Load the cleaned dataset.
- Separate features and target.
- Exclude duration to prevent potential target leakage.
- Perform stratified train-validation splitting.
- Build numerical and categorical preprocessing transformers.
- Build leakage-safe SMOTENC model pipelines.
"""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline


NUMERICAL_FEATURES = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

TARGET_COLUMN = "y"


def load_cleaned_data(data_path):
    """Load the cleaned dataset from a CSV file."""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    return pd.read_csv(data_path)


def split_features_target(data):
    """Separate input features and convert the target into binary labels."""
    required_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = data[TARGET_COLUMN].map({"no": 0, "yes": 1})

    if y.isna().any():
        raise ValueError("Target contains unexpected or missing values.")

    return X, y.astype(int)


def split_data(X, y, test_size=0.20, random_state=42):
    """Create stratified training and validation datasets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor():
    """Build the numerical and categorical preprocessing transformer."""

    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_smotenc_model_pipeline(
    estimator,
    random_state=42,
    k_neighbors=5,
):
    """
    Build a leakage-safe pipeline using SMOTENC.

    Order:
    1. SMOTENC resamples mixed raw training features.
    2. ColumnTransformer scales and encodes features.
    3. Estimator trains on the processed features.

    During prediction, the sampler is skipped automatically.
    During cross-validation, each training fold fits its own sampler.
    """

    sampler = SMOTENC(
        categorical_features=CATEGORICAL_FEATURES,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )

    return ImbPipeline(
        steps=[
            ("sampler", sampler),
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def prepare_data(data_path, test_size=0.20, random_state=42):
    """
    Load, split and preprocess the dataset.

    The transformer is fitted only on training data.
    This function is intended for preprocessing exploration.
    """

    data = load_cleaned_data(data_path)
    X, y = split_features_target(data)

    X_train, X_valid, y_train, y_valid = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "X_train_processed": X_train_processed,
        "X_valid_processed": X_valid_processed,
        "preprocessor": preprocessor,
    }