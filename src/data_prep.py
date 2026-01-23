# src/data_prep.py

import os
import shutil
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------
RAW_DATA_PATH = os.path.join("data", "raw")
RAW_FILE_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_FILE_PATH = os.path.join(RAW_DATA_PATH, RAW_FILE_NAME)

KAGGLE_DATASET_ID = "blastchar/telco-customer-churn"


# ------------------------------------------------------------
# Dataset loading (with Kaggle download if needed)
# ------------------------------------------------------------
def load_telco() -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset.
    If the raw CSV is not present locally, download it from Kaggle using kagglehub.
    """
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    if not os.path.exists(RAW_FILE_PATH):
        dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]

        if len(csv_files) != 1:
            raise ValueError(
                f"Expected exactly 1 CSV file, found {len(csv_files)}: {csv_files}"
            )

        shutil.copy(
            os.path.join(dataset_path, csv_files[0]),
            RAW_FILE_PATH
        )
        print("Dataset downloaded from Kaggle and copied to data/raw/")
    else:
        print("Dataset already present in data/raw/")

    df = pd.read_csv(RAW_FILE_PATH)

    # Stable binary target (do not overwrite df["Churn"])
    df["churn_flag"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # TotalCharges: convert to numeric (empty strings -> NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


# ------------------------------------------------------------
# Feature definitions
# ------------------------------------------------------------
def get_feature_lists():
    """Return business-driven feature lists."""
    cat_features = [
        "Contract",
        "TechSupport",
        "OnlineSecurity",
        "InternetService",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    num_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    return cat_features, num_features


# ------------------------------------------------------------
# Train / test split
# ------------------------------------------------------------
def make_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split dataset into train/test sets with stratification."""
    X = df.drop(columns=["customerID", "Churn", "churn_flag"])
    y = df["churn_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------
# Preprocessing pipeline
# ------------------------------------------------------------
def build_preprocessor(cat_features, num_features) -> ColumnTransformer:
    """Build preprocessing pipeline (imputation + scaling + encoding)."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    return preprocessor
