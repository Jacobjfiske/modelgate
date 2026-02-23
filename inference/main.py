from fastapi import FastAPI, HTTPException

from inference.config import get_runtime_config
from inference.feature_validation import FeatureValidationError, validate_features
from inference.metrics import (
    MODEL_SELECTION_FALLBACK_TOTAL,
    REQUEST_BY_MODEL_VERSION_TOTAL,
    REQUEST_ERRORS_TOTAL,
    REQUEST_LATENCY_SECONDS,
    REQUEST_SUCCESS_TOTAL,
    render_metrics,
)
from inference.model_loader import ModelLoadError, load_model_artifact
from inference.schemas import InferenceRequest, InferenceResponse
from inference.service import classify_risk, score_linear_model

app = FastAPI(title="ModelGate Inference API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return render_metrics()


@app.post("/v1/inference", response_model=InferenceResponse)
def infer(payload: InferenceRequest) -> InferenceResponse:
    runtime = get_runtime_config()
    with REQUEST_LATENCY_SECONDS.time():
        try:
            model_version, model, metadata, used_fallback = load_model_artifact(
                requested_version=runtime.model_version,
                fallback_to_stable=runtime.model_fallback_to_stable,
            )
        except ModelLoadError as exc:
            REQUEST_ERRORS_TOTAL.labels(error_type="model_load").inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if used_fallback:
            MODEL_SELECTION_FALLBACK_TOTAL.inc()

        payload_dict = payload.model_dump()
        try:
            schema = metadata.get("schema")
            validate_features(payload_dict, schema)
        except FeatureValidationError as exc:
            REQUEST_ERRORS_TOTAL.labels(error_type="feature_validation").inc()
            REQUEST_BY_MODEL_VERSION_TOTAL.labels(
                model_version=model_version,
                outcome="validation_error",
            ).inc()
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            risk_score = score_linear_model(payload_dict, model)
            risk_band = classify_risk(risk_score)
        except Exception as exc:
            REQUEST_ERRORS_TOTAL.labels(error_type="inference_runtime").inc()
            REQUEST_BY_MODEL_VERSION_TOTAL.labels(
                model_version=model_version,
                outcome="runtime_error",
            ).inc()
            raise HTTPException(status_code=500, detail="inference runtime error") from exc

    REQUEST_SUCCESS_TOTAL.inc()
    REQUEST_BY_MODEL_VERSION_TOTAL.labels(model_version=model_version, outcome="success").inc()
    return InferenceResponse(
        model_version=model_version,
        risk_score=risk_score,
        risk_band=risk_band,
    )
