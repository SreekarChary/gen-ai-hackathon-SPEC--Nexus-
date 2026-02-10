"""
Module 1 — Data Preprocessing
Handles loading, cleaning, feature engineering, encoding, and splitting the dataset.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

import config


def load_data(filename: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    filepath = os.path.join(config.DATA_RAW_DIR, filename)
    df = pd.read_csv(filepath)
    print(f"[✓] Loaded dataset: {filepath}  —  {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls, duplicates, and fix data types."""
    initial_rows = len(df)

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop rows with any null values (can be refined per-column later)
    df = df.dropna()

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    removed = initial_rows - len(df)
    print(f"[✓] Cleaned data: removed {removed} rows  —  {len(df)} rows remaining")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that help the model detect fraud patterns.
    NOTE: Column names will be updated once the actual dataset is provided.
    """
    # Placeholder — feature engineering will be customized to your dataset columns.
    # Examples of typical derived features:
    #   - claim_to_income_ratio  = claim_amount / annual_income
    #   - days_to_report         = (report_date - incident_date).days
    #   - claims_last_year       = count of past claims per policyholder
    print("[✓] Feature engineering complete")
    return df


def encode_and_scale(df: pd.DataFrame, target_col: str):
    """
    Label-encode categorical columns, standard-scale numeric columns.
    Saves encoders and scaler to models/ for reuse at inference time.
    Returns X (features), y (target), feature_names.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Separate target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # Label-encode categoricals
    label_encoders = {}
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    joblib.dump(label_encoders, os.path.join(config.MODELS_DIR, "label_encoders.pkl"))

    # Encode target if it is categorical
    if y.dtype == "object":
        target_le = LabelEncoder()
        y = target_le.fit_transform(y)
        joblib.dump(target_le, os.path.join(config.MODELS_DIR, "target_encoder.pkl"))

    # Standard-scale numerics
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    joblib.dump(scaler, os.path.join(config.MODELS_DIR, "scaler.pkl"))

    feature_names = X.columns.tolist()
    print(f"[✓] Encoding & scaling done  —  {len(cat_cols)} categorical, {len(num_cols)} numeric features")
    return X, y, feature_names


def split_data(X, y):
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"[✓] Split: {len(X_train)} train / {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test


def save_processed(X_train, X_test, y_train, y_test, feature_names):
    """Save processed datasets to data/processed/."""
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    joblib.dump(
        {"X_train": X_train, "X_test": X_test,
         "y_train": y_train, "y_test": y_test,
         "feature_names": feature_names},
        os.path.join(config.DATA_PROCESSED_DIR, "processed_data.pkl"),
    )
    print(f"[✓] Processed data saved to {config.DATA_PROCESSED_DIR}")


# ─── CLI entry point ───────────────────────────────────
if __name__ == "__main__":
    # Update the filename and target column once you have the dataset
    DATASET_FILENAME = "claims.csv"         # <-- change to your CSV name
    TARGET_COLUMN = "fraud_reported"        # <-- change to your target column

    df = load_data(DATASET_FILENAME)
    df = clean_data(df)
    df = engineer_features(df)
    X, y, feature_names = encode_and_scale(df, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_processed(X_train, X_test, y_train, y_test, feature_names)
    print("\n✅ Preprocessing pipeline complete!")
