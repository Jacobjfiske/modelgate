"""Train a baseline model and register a versioned artifact.

This script uses a tiny synthetic dataset to produce deterministic coefficients
for a linear risk score model used by the inference API.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"


def generate_dataset() -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [
            [50.0, 365, 1.2, 0.10],
            [600.0, 20, 6.0, 0.80],
            [120.0, 900, 0.8, 0.15],
            [400.0, 45, 4.5, 0.65],
            [75.0, 1200, 0.6, 0.05],
            [900.0, 10, 7.5, 0.90],
            [250.0, 180, 2.7, 0.35],
            [520.0, 50, 5.2, 0.72],
        ],
        dtype=float,
    )
    y = np.array([0.06, 0.89, 0.10, 0.73, 0.05, 0.96, 0.34, 0.80], dtype=float)
    return x, y


def fit_linear_risk_model(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    x_design = np.hstack([x, np.ones((x.shape[0], 1))])
    params, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    weights = params[:-1]
    intercept = params[-1]
    return weights, float(intercept)


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"stable": None, "canary": None, "versions": []}
    return json.loads(REGISTRY_PATH.read_text())


def save_registry(registry: dict) -> None:
    # Write then replace to avoid partial registry files.
    tmp_path = REGISTRY_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(registry, indent=2) + "\n")
    tmp_path.replace(REGISTRY_PATH)


def register_model_version(registry: dict, model_version: str, promote_stable: bool) -> dict:
    if registry.get("stable") is None:
        registry["stable"] = model_version
    elif promote_stable:
        registry["stable"] = model_version

    # New versions start in canary.
    registry["canary"] = model_version

    versions = registry.get("versions", [])
    versions.append(
        {
            "model_version": model_version,
            "artifact_path": f"models/{model_version}/model.json",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ready",
        }
    )
    registry["versions"] = versions
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline model and register versioned artifacts."
    )
    parser.add_argument(
        "--promote-stable",
        action="store_true",
        help="Promote this trained version to the stable channel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = generate_dataset()
    weights, intercept = fit_linear_risk_model(x, y)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    model_version = f"v{ts}"
    version_dir = MODELS_DIR / model_version
    suffix = 1
    # Guard against collisions in very fast repeated runs.
    while version_dir.exists():
        model_version = f"v{ts}-{suffix}"
        version_dir = MODELS_DIR / model_version
        suffix += 1
    version_dir.mkdir(parents=True, exist_ok=False)

    artifact = {
        "model_type": "linear-risk-score",
        "feature_order": [
            "transaction_amount",
            "account_age_days",
            "avg_daily_transactions",
            "country_risk_score",
        ],
        "weights": [float(v) for v in weights],
        "intercept": intercept,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata = {
        "model_version": model_version,
        "intended_stage": "staging",
        "metrics": {"mae": float(np.mean(np.abs(y - (x @ weights + intercept))))},
        "schema": {
            "transaction_amount": {"type": "float", "min": 0.0, "max": 10000.0},
            "account_age_days": {"type": "int", "min": 0, "max": 5000},
            "avg_daily_transactions": {"type": "float", "min": 0.0, "max": 200.0},
            "country_risk_score": {"type": "float", "min": 0.0, "max": 1.0},
        },
    }

    (version_dir / "model.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    registry = register_model_version(
        registry=load_registry(),
        model_version=model_version,
        promote_stable=args.promote_stable,
    )
    save_registry(registry)

    print(f"registered model version: {model_version}")


if __name__ == "__main__":
    main()
