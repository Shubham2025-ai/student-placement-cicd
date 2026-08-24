"""
Evaluates the trained model against the held-out test set and enforces
the CI/CD quality gate: pipeline must fail if accuracy < 80%.

Run as a CI step:
    python -m src.evaluate
"""

import os
import sys

import joblib
from sklearn.metrics import accuracy_score

from src.train import MODEL_PATH, TEST_DATA_PATH

ACCURACY_THRESHOLD = 0.80


def evaluate() -> float:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Required model file not found: {MODEL_PATH}. "
            "Did the training step run?"
        )
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data file not found: {TEST_DATA_PATH}")

    model = joblib.load(MODEL_PATH)
    X_test, y_test = joblib.load(TEST_DATA_PATH)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


if __name__ == "__main__":
    try:
        acc = evaluate()
    except FileNotFoundError as exc:
        print(f"Evaluation FAILED: {exc}")
        sys.exit(1)

    print(f"Model accuracy: {acc:.4f} (threshold: {ACCURACY_THRESHOLD:.2f})")

    if acc < ACCURACY_THRESHOLD:
        print("Evaluation FAILED: accuracy below required threshold")
        sys.exit(1)

    print("Evaluation PASSED: accuracy meets required threshold")
    sys.exit(0)
