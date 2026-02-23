from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

REQUEST_LATENCY_SECONDS = Histogram(
    "inference_request_latency_seconds",
    "Latency of inference requests",
)
REQUEST_ERRORS_TOTAL = Counter(
    "inference_request_errors_total",
    "Total inference request errors",
    ["error_type"],
)
REQUEST_SUCCESS_TOTAL = Counter(
    "inference_request_success_total", "Total successful inference requests"
)
REQUEST_BY_MODEL_VERSION_TOTAL = Counter(
    "inference_request_total",
    "Inference requests by model version and outcome",
    ["model_version", "outcome"],
)
MODEL_SELECTION_FALLBACK_TOTAL = Counter(
    "model_selection_fallback_total",
    "Times requested model version was missing and service fell back to stable",
)


def render_metrics() -> Response:
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
