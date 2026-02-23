import json
from pathlib import Path

from fastapi.testclient import TestClient

from inference.main import app

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


def seed_model_fixture(model_version: str = "vtest-001") -> None:
    model_dir = MODELS_DIR / model_version
    model_dir.mkdir(parents=True, exist_ok=True)

    model = {
        "model_type": "linear-risk-score",
        "feature_order": [
            "transaction_amount",
            "account_age_days",
            "avg_daily_transactions",
            "country_risk_score",
        ],
        "weights": [0.0005, -0.0001, 0.03, 0.45],
        "intercept": 0.02,
    }
    metadata = {
        "model_version": model_version,
        "schema": {
            "transaction_amount": {"type": "float", "min": 0.0, "max": 10000.0},
            "account_age_days": {"type": "int", "min": 0, "max": 5000},
            "avg_daily_transactions": {"type": "float", "min": 0.0, "max": 200.0},
            "country_risk_score": {"type": "float", "min": 0.0, "max": 1.0},
        },
    }
    registry = {"stable": model_version, "canary": model_version, "versions": []}

    (model_dir / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (MODELS_DIR / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")


def test_inference_endpoint_returns_model_version(monkeypatch):
    seed_model_fixture()
    monkeypatch.setenv("MODEL_VERSION", "stable")

    client = TestClient(app)
    payload = {
        "transaction_amount": 120.0,
        "account_age_days": 400,
        "avg_daily_transactions": 1.3,
        "country_risk_score": 0.2,
    }

    response = client.post("/v1/inference", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["model_version"] == "vtest-001"
    assert 0.0 <= body["risk_score"] <= 1.0


def test_inference_rejects_invalid_feature_range(monkeypatch):
    seed_model_fixture(model_version="vtest-002")
    monkeypatch.setenv("MODEL_VERSION", "stable")

    client = TestClient(app)
    payload = {
        "transaction_amount": 120.0,
        "account_age_days": 400,
        "avg_daily_transactions": 1.3,
        "country_risk_score": 2.0,
    }

    response = client.post("/v1/inference", json=payload)
    assert response.status_code == 422


def test_inference_falls_back_to_stable_when_enabled(monkeypatch):
    seed_model_fixture(model_version="vtest-003")
    monkeypatch.setenv("MODEL_VERSION", "vmissing-999")
    monkeypatch.setenv("MODEL_FALLBACK_TO_STABLE", "true")

    client = TestClient(app)
    payload = {
        "transaction_amount": 120.0,
        "account_age_days": 400,
        "avg_daily_transactions": 1.3,
        "country_risk_score": 0.2,
    }

    response = client.post("/v1/inference", json=payload)
    assert response.status_code == 200
    assert response.json()["model_version"] == "vtest-003"


def test_inference_returns_503_for_invalid_model_metadata(monkeypatch):
    seed_model_fixture(model_version="vtest-004")
    monkeypatch.setenv("MODEL_VERSION", "stable")
    monkeypatch.delenv("MODEL_FALLBACK_TO_STABLE", raising=False)

    model_dir = MODELS_DIR / "vtest-004"
    bad_metadata = {"model_version": "vtest-004"}
    (model_dir / "metadata.json").write_text(json.dumps(bad_metadata, indent=2) + "\n")

    client = TestClient(app)
    payload = {
        "transaction_amount": 120.0,
        "account_age_days": 400,
        "avg_daily_transactions": 1.3,
        "country_risk_score": 0.2,
    }

    response = client.post("/v1/inference", json=payload)
    assert response.status_code == 503
