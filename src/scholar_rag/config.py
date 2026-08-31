"""Central configuration, loaded from environment / .env (no hard dependency on
pydantic-settings so the module stays import-light)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Project root = scholar-rag/  (…/src/scholar_rag/config.py -> parents[2]).
# Lets the app find .env and data/ regardless of the launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:  # optional convenience; app still works with real env vars if absent
    from dotenv import load_dotenv

    _env_file = PROJECT_ROOT / ".env"
    load_dotenv(_env_file if _env_file.exists() else None)
except Exception:  # pragma: no cover
    pass


def _resolve_data_dir(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


# provider -> (default base_url, api-key env var, default model)
PROVIDER_DEFAULTS: dict[str, tuple[str | None, str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "openai": (None, "OPENAI_API_KEY", "gpt-4o-mini"),
    "custom": (None, "LLM_API_KEY", "llama3.2:3b"),
}


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    corpus_confidence_threshold: float = 0.35
    # Storage
    vector_backend: str = "auto"
    data_dir: Path = Path("data")

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def numpy_store_path(self) -> Path:
        return self.data_dir / "numpy_store"

    @property
    def has_llm_key(self) -> bool:
        # custom/local servers legitimately need no key
        return bool(self.llm_api_key) or self.llm_provider == "custom"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    provider = (_get("LLM_PROVIDER", "groq") or "groq").lower()
    base_default, key_env, model_default = PROVIDER_DEFAULTS.get(
        provider, PROVIDER_DEFAULTS["groq"]
    )
    return Settings(
        llm_provider=provider,
        llm_model=_get("LLM_MODEL", model_default),
        llm_base_url=_get("LLM_BASE_URL", base_default),
        llm_api_key=_get("LLM_API_KEY") or _get(key_env),
        llm_temperature=float(_get("LLM_TEMPERATURE", "0.1")),
        llm_max_tokens=int(_get("LLM_MAX_TOKENS", "1024")),
        embedding_model=_get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size=int(_get("CHUNK_SIZE", "1000")),
        chunk_overlap=int(_get("CHUNK_OVERLAP", "150")),
        top_k=int(_get("TOP_K", "5")),
        corpus_confidence_threshold=float(_get("CORPUS_CONFIDENCE_THRESHOLD", "0.35")),
        vector_backend=(_get("VECTOR_BACKEND", "auto") or "auto").lower(),
        data_dir=_resolve_data_dir(_get("DATA_DIR", "data")),
    )
