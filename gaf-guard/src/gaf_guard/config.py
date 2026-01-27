from functools import lru_cache

from pydantic import HttpUrl, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Configuration(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_nested_delimiter="__",
        extra="allow",
    )

    GAF_GUARD_HOST: str = "localhost"
    GAF_GUARD_PORT: int = 8000
    OLLAMA_API_URL: HttpUrl = HttpUrl("http://localhost:11434")
    RITS_API_KEY: str = "cbc683b3a1a7c52d2a73008b785d2811"
    RITS_API_URL: str = (
        "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com"
    )


@lru_cache
def get_configuration() -> Configuration:
    """Get cached configuration"""
    try:
        return Configuration()
    except ValidationError as ex:
        raise ValueError(
            "Improperly configured, make sure to supply all required variables"
        ) from ex
