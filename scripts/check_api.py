"""
Starts the Flask app in-process (no real network server needed) and sends
a real request through its test client to confirm the prediction API is
working end-to-end. Used as a CI step, separate from the pytest API tests,
to explicitly gate deployment on "Prediction API fails".

Run as a CI step:
    python scripts/check_api.py
"""

import sys

sys.path.insert(0, ".")

from app import app  # noqa: E402


def main() -> int:
    client = app.test_client()

    health = client.get("/health")
    if health.status_code != 200:
        print(f"API check FAILED: /health returned {health.status_code}")
        return 1

    payload = {
        "cgpa": 8.5,
        "attendance": 90,
        "coding_score": 75,
        "projects": 3,
        "internships": 1,
        "communication_score": 80,
    }
    resp = client.post("/predict", json=payload)
    if resp.status_code != 200:
        print(f"API check FAILED: /predict returned {resp.status_code}")
        return 1

    data = resp.get_json()
    if "prediction" not in data or "placed" not in data:
        print(f"API check FAILED: unexpected response shape: {data}")
        return 1

    print(f"API check PASSED: /predict responded with {data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
