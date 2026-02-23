from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    model_version: str = Field(default="stable", alias="MODEL_VERSION")
    model_fallback_to_stable: bool = Field(default=False, alias="MODEL_FALLBACK_TO_STABLE")
    service_name: str = Field(default="modelgate-inference", alias="SERVICE_NAME")


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    model_fallback_to_stable: bool
    service_name: str


def get_runtime_config() -> RuntimeConfig:
    s = Settings()
    return RuntimeConfig(
        model_version=s.model_version,
        model_fallback_to_stable=s.model_fallback_to_stable,
        service_name=s.service_name,
    )
