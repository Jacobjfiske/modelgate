from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"


class ModelLoadError(RuntimeError):
    pass


def _model_version_exists(registry: dict[str, Any], model_version: str) -> bool:
    versions = registry.get("versions", [])
    return any(item.get("model_version") == model_version for item in versions)


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        raise ModelLoadError("model registry is missing")
    return json.loads(REGISTRY_PATH.read_text())


def resolve_model_version(
    requested_version: str, fallback_to_stable: bool = False
) -> tuple[str, bool]:
    registry = load_registry()
    # Channel names resolve to the current pointer in the registry.
    if requested_version in {"stable", "canary"}:
        resolved = registry.get(requested_version)
        if not resolved:
            raise ModelLoadError(f"channel '{requested_version}' is not set")
        return resolved, False

    if _model_version_exists(registry, requested_version):
        return requested_version, False

    if fallback_to_stable:
        stable = registry.get("stable")
        if not stable:
            raise ModelLoadError(
                f"requested version '{requested_version}' missing and stable channel is not set"
            )
        return stable, True

    raise ModelLoadError(f"requested model version '{requested_version}' is not registered")


def validate_model_contract(
    model: dict[str, Any], metadata: dict[str, Any], model_version: str
) -> None:
    feature_order = model.get("feature_order")
    weights = model.get("weights")
    if not isinstance(feature_order, list) or not feature_order:
        raise ModelLoadError(f"model '{model_version}' has invalid feature_order")
    if not isinstance(weights, list) or len(weights) != len(feature_order):
        raise ModelLoadError(f"model '{model_version}' has mismatched weights and feature_order")
    if "intercept" not in model:
        raise ModelLoadError(f"model '{model_version}' missing intercept")

    schema = metadata.get("schema")
    if not isinstance(schema, dict) or not schema:
        raise ModelLoadError(f"model '{model_version}' metadata missing schema")

    schema_keys = set(schema.keys())
    if set(feature_order) != schema_keys:
        raise ModelLoadError(f"model '{model_version}' schema/features mismatch")


def load_model_artifact(
    requested_version: str, fallback_to_stable: bool = False
) -> tuple[str, dict[str, Any], dict[str, Any], bool]:
    model_version, used_fallback = resolve_model_version(
        requested_version=requested_version, fallback_to_stable=fallback_to_stable
    )
    model_dir = MODELS_DIR / model_version
    model_path = model_dir / "model.json"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        raise ModelLoadError(f"model artifact not found for version '{model_version}'")

    model = json.loads(model_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    # Validate artifacts before serving traffic.
    validate_model_contract(model=model, metadata=metadata, model_version=model_version)
    return model_version, model, metadata, used_fallback
