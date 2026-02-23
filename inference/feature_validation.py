from __future__ import annotations

from typing import Any


class FeatureValidationError(ValueError):
    pass


def validate_features(payload: dict[str, Any], schema: dict[str, dict[str, Any]]) -> None:
    if not isinstance(schema, dict) or not schema:
        raise FeatureValidationError("invalid or empty feature schema in model metadata")

    expected = set(schema.keys())
    received = set(payload.keys())

    missing = sorted(expected - received)
    unexpected = sorted(received - expected)

    if missing:
        raise FeatureValidationError(f"missing required features: {', '.join(missing)}")
    if unexpected:
        raise FeatureValidationError(f"unexpected features: {', '.join(unexpected)}")

    for key, rules in schema.items():
        if not isinstance(rules, dict):
            raise FeatureValidationError(f"feature '{key}' rules must be a mapping")

        value = payload[key]
        expected_type = rules.get("type")
        if expected_type == "int" and isinstance(value, bool):
            raise FeatureValidationError(f"feature '{key}' must be int")
        if expected_type == "int" and not isinstance(value, int):
            raise FeatureValidationError(f"feature '{key}' must be int")
        if expected_type == "float" and not isinstance(value, (int, float)):
            raise FeatureValidationError(f"feature '{key}' must be float")

        min_v = rules.get("min")
        max_v = rules.get("max")
        if min_v is not None and value < min_v:
            raise FeatureValidationError(f"feature '{key}' below minimum ({min_v})")
        if max_v is not None and value > max_v:
            raise FeatureValidationError(f"feature '{key}' above maximum ({max_v})")
