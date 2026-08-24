"""
Preprocessing: splits the validated dataset into train/test feature and
label sets, ready for training.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "cgpa",
    "attendance",
    "coding_score",
    "projects",
    "internships",
    "communication_score",
]
TARGET_COLUMN = "placed"


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature matrix X and label vector y."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def train_test_split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """Return X_train, X_test, y_train, y_test for the given dataframe."""
    X, y = split_features_labels(df)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
