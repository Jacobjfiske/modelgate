# Model Artifact Convention

Every model is stored as `models/<model_version>/`.

Required files:
- `model.json`: model parameters used by inference runtime.
- `metadata.json`: schema expectations and training metadata.

`models/registry.json` tracks channel pointers:
- `stable`: default production model.
- `canary`: optional candidate for controlled rollout.
- `versions`: immutable version history.
