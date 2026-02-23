import json

import pytest

from inference import model_loader
from inference.model_loader import ModelLoadError, load_model_artifact


def test_load_model_artifact_fallback_to_stable(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    model_version = "vstable-001"
    version_dir = models_dir / model_version
    version_dir.mkdir(parents=True)

    model = {
        "feature_order": ["transaction_amount"],
        "weights": [0.01],
        "intercept": 0.0,
    }
    metadata = {
        "schema": {
            "transaction_amount": {"type": "float", "min": 0.0, "max": 10000.0}
        }
    }
    registry = {
        "stable": model_version,
        "canary": model_version,
        "versions": [{"model_version": model_version}],
    }

    (version_dir / "model.json").write_text(json.dumps(model))
    (version_dir / "metadata.json").write_text(json.dumps(metadata))
    (models_dir / "registry.json").write_text(json.dumps(registry))

    monkeypatch.setattr(model_loader, "MODELS_DIR", models_dir)
    monkeypatch.setattr(model_loader, "REGISTRY_PATH", models_dir / "registry.json")

    resolved_version, _, _, used_fallback = load_model_artifact(
        requested_version="vmissing-123", fallback_to_stable=True
    )

    assert resolved_version == model_version
    assert used_fallback is True


def test_load_model_artifact_rejects_missing_registration(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    registry = {"stable": None, "canary": None, "versions": []}

    (models_dir / "registry.json").write_text(json.dumps(registry))

    monkeypatch.setattr(model_loader, "MODELS_DIR", models_dir)
    monkeypatch.setattr(model_loader, "REGISTRY_PATH", models_dir / "registry.json")

    with pytest.raises(ModelLoadError):
        load_model_artifact(requested_version="vmissing-123", fallback_to_stable=False)


def test_load_model_artifact_rejects_contract_mismatch(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    model_version = "vcontract-001"
    model_dir = models_dir / model_version
    model_dir.mkdir(parents=True)

    model = {
        "feature_order": ["transaction_amount", "account_age_days"],
        "weights": [0.5],
        "intercept": 0.0,
    }
    metadata = {
        "schema": {
            "transaction_amount": {"type": "float", "min": 0.0, "max": 10000.0}
        }
    }
    registry = {
        "stable": model_version,
        "canary": model_version,
        "versions": [{"model_version": model_version}],
    }

    (model_dir / "model.json").write_text(json.dumps(model))
    (model_dir / "metadata.json").write_text(json.dumps(metadata))
    (models_dir / "registry.json").write_text(json.dumps(registry))

    monkeypatch.setattr(model_loader, "MODELS_DIR", models_dir)
    monkeypatch.setattr(model_loader, "REGISTRY_PATH", models_dir / "registry.json")

    with pytest.raises(ModelLoadError):
        load_model_artifact(requested_version="stable", fallback_to_stable=False)
