# Rollback Notes

1. Set `MODEL_VERSION` to the last known-good version in deployment config.
2. Disable canary routing by setting `CANARY_ENABLED=false`.
3. Redeploy staging first, then production.
4. Validate `/health`, `/metrics`, and a smoke inference request.
5. Record incident timeline and root cause in runbook.
