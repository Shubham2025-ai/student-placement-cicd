import os

from src.evaluate import evaluate
from src.train import MODEL_PATH, TEST_DATA_PATH, train


def test_train_creates_model_file():
    path = train()
    assert path == MODEL_PATH
    assert os.path.exists(MODEL_PATH)
    assert os.path.exists(TEST_DATA_PATH)


def test_model_has_predict_method():
    import joblib

    train()
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")


def test_evaluate_reports_accuracy_above_threshold():
    train()
    accuracy = evaluate()
    assert 0.0 <= accuracy <= 1.0
    assert accuracy >= 0.80
