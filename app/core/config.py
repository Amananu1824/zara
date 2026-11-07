"""Central configuration loaded from the project's .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from None


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number; got {raw!r}") from None

def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    value = raw.strip().casefold()

    if value in {"1", "true", "yes", "on"}:
        return True

    if value in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be a boolean value."
    )

@dataclass(frozen=True)
class Settings:
    # Ollama
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ).rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b").strip()
    ollama_timeout_seconds: float = _float_env(
        "OLLAMA_TIMEOUT_SECONDS",
        120.0,
    )
    ollama_temperature: float = _float_env("OLLAMA_TEMPERATURE", 0.6)
    ollama_top_p: float = _float_env("OLLAMA_TOP_P", 0.9)
    ollama_repeat_penalty: float = _float_env(
        "OLLAMA_REPEAT_PENALTY",
        1.10,
    )

    # Conversation and API
    api_allow_database_tools: bool = _bool_env(
        "ZARA_API_ALLOW_DATABASE_TOOLS",
        False,
    )

    api_allow_database_mutations: bool = _bool_env(
        "ZARA_API_ALLOW_DATABASE_MUTATIONS",
        False,
    )
    max_history_messages: int = _int_env("MAX_HISTORY_MESSAGES", 12)
    max_input_chars: int = _int_env("MAX_INPUT_CHARS", 12000)
    api_host: str = os.getenv("ZARA_API_HOST", "127.0.0.1")
    api_port: int = _int_env("ZARA_API_PORT", 8000)

    api_allow_system_control_tools: bool = _bool_env(
        "ZARA_API_ALLOW_SYSTEM_CONTROL_TOOLS",
        False,
    )

    api_allow_system_control_mutations: bool = _bool_env(
        "ZARA_API_ALLOW_SYSTEM_CONTROL_MUTATIONS",
        False,
    )
    
    # PostgreSQL
    zara_database_host: str = os.getenv(
        "ZARA_DATABASE_HOST",
        "127.0.0.1",
    ).strip()
    zara_database_port: int = _int_env("ZARA_DATABASE_PORT", 5432)
    zara_database_name: str = os.getenv(
        "ZARA_DATABASE_NAME",
        "zara",
    ).strip()
    zara_database_user: str = os.getenv(
        "ZARA_DATABASE_USER",
        "zara_app",
    ).strip()
    zara_database_password: str = os.getenv(
        "ZARA_DATABASE_PASSWORD",
        "",
    )
    zara_database_connect_timeout: int = _int_env(
        "ZARA_DATABASE_CONNECT_TIMEOUT",
        5,
    )
    zara_database_statement_timeout_ms: int = _int_env(
        "ZARA_DATABASE_STATEMENT_TIMEOUT_MS",
        30000,
    )

    # Embeddings. These settings prepare the schema; the model is
    # not called during ordinary database initialization.
    zara_embedding_model: str = os.getenv(
        "ZARA_EMBEDDING_MODEL",
        "embeddinggemma",
    ).strip()
    zara_embedding_model_version: str = os.getenv(
        "ZARA_EMBEDDING_MODEL_VERSION",
        "unverified",
    ).strip() or "unverified"
    zara_embedding_dimensions: int = _int_env(
        "ZARA_EMBEDDING_DIMENSIONS",
        768,
    )


settings = Settings()

if not settings.ollama_model:
    raise ValueError("OLLAMA_MODEL cannot be empty. Set it in the .env file.")
if settings.ollama_timeout_seconds <= 0:
    raise ValueError("OLLAMA_TIMEOUT_SECONDS must be greater than zero.")
if settings.max_history_messages < 2:
    raise ValueError("MAX_HISTORY_MESSAGES must be at least 2.")
if settings.max_input_chars < 100:
    raise ValueError("MAX_INPUT_CHARS must be at least 100.")

if not settings.zara_database_host:
    raise ValueError("ZARA_DATABASE_HOST cannot be empty.")
if not settings.zara_database_name:
    raise ValueError("ZARA_DATABASE_NAME cannot be empty.")
if not settings.zara_database_user:
    raise ValueError("ZARA_DATABASE_USER cannot be empty.")
if not 1 <= settings.zara_database_port <= 65535:
    raise ValueError("ZARA_DATABASE_PORT must be between 1 and 65535.")
if settings.zara_database_connect_timeout <= 0:
    raise ValueError("ZARA_DATABASE_CONNECT_TIMEOUT must be positive.")
if not 1_000 <= settings.zara_database_statement_timeout_ms <= 300_000:
    raise ValueError(
        "ZARA_DATABASE_STATEMENT_TIMEOUT_MS must be between 1000 and 300000."
    )
if not settings.zara_embedding_model:
    raise ValueError("ZARA_EMBEDDING_MODEL cannot be empty.")
if len(settings.zara_embedding_model_version) > 120:
    raise ValueError(
        "ZARA_EMBEDDING_MODEL_VERSION cannot exceed 120 characters."
    )
if not 1 <= settings.zara_embedding_dimensions <= 2000:
    raise ValueError(
        "ZARA_EMBEDDING_DIMENSIONS must be between 1 and 2000 "
        "for the planned vector index."
    )
if not 0.5 <= settings.ollama_repeat_penalty <= 2.0:
    raise ValueError(
        "OLLAMA_REPEAT_PENALTY must be between 0.5 and 2.0."
    )