from functools import lru_cache
from typing import Optional

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
    RITS_API_KEY: Optional[str] = ""
    RITS_API_URL: Optional[str] = ""
    OLLAMA_API_URL: Optional[str] = ""


@lru_cache
def get_configuration() -> Configuration:
    """Get cached configuration"""
    try:
        return Configuration()
    except ValidationError as ex:
        raise ValueError(
            "Improperly configured, make sure to supply all required variables"
        ) from ex
