"""Configuration management for benchmark metadata processing."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration settings for the benchmark processing pipeline."""

    # Path Configuration (using pathlib for better path handling)
    # From config.py: src/benchmarkcard/config.py → 2x parent = project root
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    EXTERNAL_DIR: Path = PROJECT_ROOT / "external"
    FACTREASONER_DIR: Path = EXTERNAL_DIR / "FactReasoner"
    MERLIN_BIN: Path = EXTERNAL_DIR / "merlin" / "bin" / "merlin"

    # LLM Configuration — tiered models for different task complexities
    COMPOSER_MODEL: str = os.getenv(
        "RITS_COMPOSER_MODEL", "deepseek-ai/DeepSeek-V3.2"
    )
    LIGHT_MODEL: str = os.getenv(
        "RITS_LIGHT_MODEL", "ibm-granite/granite-3.3-8b-instruct"
    )
    FACTREASONER_MODEL: str = "llama-3.3-70b-instruct"  # must match FactReasoner models.yaml key
    DEFAULT_MODEL: str = FACTREASONER_MODEL  # backward compat
    DEFAULT_EMBEDDING_MODEL: str = "bge-large"
    LLM_ENGINE_TYPE: str = "rits"  # or ollama, vllm

    # Processing Configuration
    DEFAULT_FACTUALITY_THRESHOLD: float = 0.8
    DEFAULT_TOP_K: int = 4

    # RAG Configuration
    ENABLE_LLM_RERANKING: bool = True
    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_QUERY_EXPANSION: bool = True

    # Chunking Configuration
    PARENT_CHUNK_SIZE: int = 2048
    CHILD_CHUNK_SIZE: int = 512

    # Directory Configuration (string-based for backward compatibility)
    FACTREASONER_CACHE_DIR: str = "factreasoner_cache"
    MERLIN_PATH: str = "external/merlin/bin/merlin"  # Deprecated: use MERLIN_BIN

    # File Extensions
    JSON_EXTENSION: str = ".json"
    JSONL_EXTENSION: str = ".jsonl"

    # Timestamp Format
    TIMESTAMP_FORMAT: str = "%Y-%m-%d_%H-%M"

    # Output Directories
    TOOL_OUTPUT_DIR: str = "tool_output"
    BENCHMARK_CARD_DIR: str = "benchmarkcard"
    OUTPUT_DIR: str = "output"

    @classmethod
    def get_env_var(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable with optional default.

        Args:
            key: Environment variable name.
            default: Default value if variable is not set.

        Returns:
            Environment variable value or default.
        """
        return os.getenv(key, default)

    @classmethod
    def validate_config(cls) -> None:
        """Validate required configuration settings.

        Raises:
            ValueError: If required environment variables are missing.
        """
        required_env_vars = ["RITS_API_KEY", "RITS_MODEL", "RITS_API_URL"]

        missing_vars = [var for var in required_env_vars if not cls.get_env_var(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


_llm_cache: dict = {}


def get_llm_handler(model_name: Optional[str] = None):
    """Get or create a cached LLM handler for the given model.

    Args:
        model_name: Model identifier (defaults to COMPOSER_MODEL).

    Returns:
        LLMHandler: Initialized LLM handler for the requested model.
    """
    import logging

    from auto_benchmarkcard.llm_handler import LLMHandler

    _log = logging.getLogger(__name__)

    key = model_name or Config.COMPOSER_MODEL
    if key not in _llm_cache:
        try:
            _log.info("Initializing LLM handler: %s", key)
            _llm_cache[key] = LLMHandler(
                engine_type=Config.LLM_ENGINE_TYPE,
                model_name=key,
                verbose=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM handler ({key}): {e}") from e
    return _llm_cache[key]


def get_light_llm_handler():
    """Get the lightweight model handler (for reranking, reformulation, atomization).

    Falls back to the composer model if the light model endpoint is unreachable.
    """
    import logging

    _log = logging.getLogger(__name__)
    try:
        return get_llm_handler(Config.LIGHT_MODEL)
    except RuntimeError:
        _log.warning(
            "Light model %s unavailable, falling back to composer model",
            Config.LIGHT_MODEL,
        )
        return get_llm_handler(Config.COMPOSER_MODEL)


class _LazyLLM:
    """Lazy proxy so importing config doesn't immediately connect to the LLM endpoint."""

    def __getattr__(self, name):
        global LLM
        LLM = get_llm_handler()
        return getattr(LLM, name)


# Backward compatibility — default LLM is the composer (heavy) model, initialized lazily
LLM = _LazyLLM()
