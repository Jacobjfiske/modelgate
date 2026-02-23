# ModelGate

Canary first ML release safety service with model contract checks and observable inference.

## Stack
- Python
- FastAPI
- Pydantic
- pytest
- Ruff
- GitHub Actions

## What it does
- Trains baseline artifacts and versions them in a local registry.
- Separates canary and stable promotion paths.
- Serves inference through an API with request validation.
- Validates runtime feature schema against artifact metadata.
- Exposes request latency, success, and error metrics.

## Entrypoints
- `python train/train_baseline.py`
- `uvicorn inference.main:app --reload --port 8010`

## Architecture
- `train/`: baseline training script that emits versioned artifacts.
- `models/`: immutable artifact folders and `registry.json` channel pointers.
- `inference/`: FastAPI scoring service with schema guard and metrics.
- `infra/`: deployment manifest template + rollback notes.
- `.github/workflows/`: CI and deployment workflow skeletons.

Reference diagram: `docs/architecture.md`.

## Model Version Flow
1. Run `python train/train_baseline.py`.
2. Script creates `models/v<timestamp>/model.json` and `metadata.json`.
3. Script updates `models/registry.json`; `canary` points to the newest trained model and `stable` changes only with `--promote-stable`.
4. Inference resolves `MODEL_VERSION` (channel or explicit version) at request time.
5. Optional fallback: set `MODEL_FALLBACK_TO_STABLE=true` to serve `stable` when a requested explicit version is missing.

## Release Flow
1. Open PR and pass CI (`ruff`, unit tests, integration tests).
2. Trigger `deploy-skeleton` to `staging` with candidate model version.
3. Validate health/metrics and run smoke tests.
4. Deploy `production` with `canary_enabled=true` for partial exposure.
5. Promote to stable rollout or rollback using `rollback_to` input.

## Reliability Controls
- Strict request schema validation via Pydantic.
- Feature validation guard against training metadata schema before scoring.
- Runtime model channel resolution (`stable`/`canary`) for fast toggles.
- Prometheus metrics surface:
  - `inference_request_latency_seconds`
  - `inference_request_errors_total{error_type=...}`
  - `inference_request_success_total`
- Rollback procedure documented in `infra/rollback-notes.md`.

## Run Locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train/train_baseline.py
# promote new artifact to stable only after validation:
python train/train_baseline.py --promote-stable
uvicorn inference.main:app --reload --port 8010
```

Inference request:
```bash
curl -X POST http://localhost:8010/v1/inference \
  -H 'Content-Type: application/json' \
  -d '{
    "transaction_amount": 120.0,
    "account_age_days": 365,
    "avg_daily_transactions": 1.5,
    "country_risk_score": 0.2
  }'
```

## Test
```bash
ruff check .
pytest -q tests/unit
pytest -q tests/integration
```

Verified test run: 14 passed on Python 3.12 container.
```bash
docker run --rm -it \
  -v "$PWD":/app \
  -w /app \
  python:3.12-slim \
  sh -lc "python -m pip install --upgrade pip && pip install -r requirements.txt && pytest -q"
```

## Current Limits
- Deployment workflow is a scaffold with placeholder commands.
- Model registry is file-based and local to repository.
- No auth/rate limiting in inference service for portfolio scope.
- No drift detector yet; schema guard is first-line protection only.

## Naming
- Repository and project name: `modelgate` / `ModelGate`
