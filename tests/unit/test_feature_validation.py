import pytest

from inference.feature_validation import FeatureValidationError, validate_features

SCHEMA = {
    "transaction_amount": {"min": 0.0, "max": 10000.0},
    "account_age_days": {"min": 0, "max": 5000},
}


def test_validate_features_accepts_matching_payload():
    payload = {"transaction_amount": 10.0, "account_age_days": 30}
    validate_features(payload, SCHEMA)


def test_validate_features_rejects_unexpected_feature():
    payload = {"transaction_amount": 10.0, "account_age_days": 30, "bad": 1}
    with pytest.raises(FeatureValidationError):
        validate_features(payload, SCHEMA)


def test_validate_features_rejects_type_mismatch():
    typed_schema = {
        "transaction_amount": {"type": "float", "min": 0.0, "max": 10000.0},
        "account_age_days": {"type": "int", "min": 0, "max": 5000},
    }
    payload = {"transaction_amount": "10.0", "account_age_days": 30}
    with pytest.raises(FeatureValidationError):
        validate_features(payload, typed_schema)
