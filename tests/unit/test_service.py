from inference.service import classify_risk, score_linear_model


def test_classify_risk_thresholds():
    assert classify_risk(0.1) == "low"
    assert classify_risk(0.4) == "medium"
    assert classify_risk(0.9) == "high"


def test_score_linear_model_range_clamped():
    model = {
        "feature_order": ["transaction_amount"],
        "weights": [100.0],
        "intercept": 0.0,
    }
    score = score_linear_model({"transaction_amount": 1.0}, model)
    assert score == 1.0
