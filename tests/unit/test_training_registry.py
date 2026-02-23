from train.train_baseline import register_model_version


def test_register_model_version_does_not_promote_stable_without_flag():
    registry = {
        "stable": "vold-stable",
        "canary": "vold-canary",
        "versions": [{"model_version": "vold-stable"}],
    }

    updated = register_model_version(registry, model_version="vnew-001", promote_stable=False)

    assert updated["stable"] == "vold-stable"
    assert updated["canary"] == "vnew-001"
    assert any(item["model_version"] == "vnew-001" for item in updated["versions"])


def test_register_model_version_promotes_when_flag_enabled():
    registry = {
        "stable": "vold-stable",
        "canary": "vold-canary",
        "versions": [{"model_version": "vold-stable"}],
    }

    updated = register_model_version(registry, model_version="vnew-002", promote_stable=True)

    assert updated["stable"] == "vnew-002"
    assert updated["canary"] == "vnew-002"
