"""
Flask prediction API for the Student Placement Prediction model.

Endpoints:
  GET  /health   -> simple liveness check
  POST /predict  -> predict placement for one student

Run locally:
    python app.py

Example request:
    curl -X POST http://127.0.0.1:5000/predict \
      -H "Content-Type: application/json" \
      -d '{"cgpa": 8.5, "attendance": 90, "coding_score": 75, \
           "projects": 3, "internships": 1, "communication_score": 80}'
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.preprocess import FEATURE_COLUMNS
from src.train import MODEL_PATH

app = Flask(__name__)

_model = None


def get_model():
    """Lazily load the trained model so importing this module never fails
    just because a model hasn't been trained yet (useful for tests)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Run `python -m src.train` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [c for c in FEATURE_COLUMNS if c not in payload]
    if missing:
        return jsonify({"error": f"Missing required field(s): {missing}"}), 400

    try:
        model = get_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    try:
        features = pd.DataFrame([{col: payload[col] for col in FEATURE_COLUMNS}])
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input values: {exc}"}), 400

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][prediction])

    result = "Placed" if prediction == 1 else "Not Placed"
    return (
        jsonify(
            {
                "prediction": result,
                "placed": prediction,
                "confidence": round(probability, 4),
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
