from __future__ import annotations

from typing import Any


def score_linear_model(payload: dict[str, float], model: dict[str, Any]) -> float:
    feature_order = model["feature_order"]
    weights = model["weights"]
    intercept = model["intercept"]

    score = intercept
    for feature_name, weight in zip(feature_order, weights, strict=True):
        score += payload[feature_name] * weight

    return max(0.0, min(1.0, float(score)))


def classify_risk(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"
