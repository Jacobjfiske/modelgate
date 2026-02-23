from pydantic import BaseModel, ConfigDict, Field


class InferenceRequest(BaseModel):
    transaction_amount: float = Field(ge=0.0, le=10000.0)
    account_age_days: int = Field(ge=0, le=5000)
    avg_daily_transactions: float = Field(ge=0.0, le=200.0)
    country_risk_score: float = Field(ge=0.0, le=1.0)


class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    risk_score: float
    risk_band: str
