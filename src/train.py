"""
Trains the placement prediction model and saves it to models/model.pkl.
Also saves the held-out test split to models/test_data.pkl so evaluate.py
can score the exact same model on the exact same test set.

Run as a CI step:
    python -m src.train
"""

import sys

import joblib
from sklearn.ensemble import RandomForestClassifier

from src.data_validation import DataValidationError, validate_file
from src.preprocess import train_test_split_data

MODEL_PATH = "models/model.pkl"
TEST_DATA_PATH = "models/test_data.pkl"


def train(data_path: str = "data/students.csv") -> str:
    df = validate_file(data_path)

    X_train, X_test, y_train, y_test = train_test_split_data(df)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42
    )
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump((X_test, y_test), TEST_DATA_PATH)

    return MODEL_PATH


if __name__ == "__main__":
    try:
        path = train()
        print(f"Model trained and saved to {path}")
        sys.exit(0)
    except (DataValidationError, FileNotFoundError) as exc:
        print(f"Training aborted, data invalid: {exc}")
        sys.exit(1)
