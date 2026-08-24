import pytest

from src.train import train
from app import app


@pytest.fixture(scope="module", autouse=True)
def trained_model():
    # Make sure a model file exists before the API tests run.
    train()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_predict_valid_input(client):
    payload = {
        "cgpa": 8.5,
        "attendance": 90,
        "coding_score": 75,
        "projects": 3,
        "internships": 1,
        "communication_score": 80,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["prediction"] in ("Placed", "Not Placed")
    assert data["placed"] in (0, 1)
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_missing_field(client):
    payload = {"cgpa": 8.5, "attendance": 90}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_no_body(client):
    resp = client.post("/predict")
    assert resp.status_code == 400
