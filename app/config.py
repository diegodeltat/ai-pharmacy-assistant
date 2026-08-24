"""Configuración central obtenida desde variables de entorno."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _as_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _as_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on", "si", "sí"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_collection: str
    chat_model: str
    embedding_model: str
    embedding_dimensions: int
    rag_top_k: int
    rag_score_threshold: float
    rag_rerank_enabled: bool
    minsal_timeout_seconds: float
    minsal_cache_seconds: int
    minsal_fallback_path: Path
    dataset_path: Path
    frontend_api_url: str

    @property
    def rag_configured(self) -> bool:
        return bool(
            self.openai_api_key
            and self.qdrant_url
            and self.qdrant_api_key
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        qdrant_collection=os.getenv(
            "QDRANT_COLLECTION",
            "drug_information_v1",
        ),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-large",
        ),
        embedding_dimensions=_as_int("EMBEDDING_DIMENSIONS", 256),
        rag_top_k=_as_int("RAG_TOP_K", 4),
        rag_score_threshold=_as_float("RAG_SCORE_THRESHOLD", 0.45),
        rag_rerank_enabled=_as_bool("RAG_RERANK_ENABLED", False),
        minsal_timeout_seconds=_as_float("MINSAL_TIMEOUT_SECONDS", 5.0),
        minsal_cache_seconds=_as_int("MINSAL_CACHE_SECONDS", 900),
        minsal_fallback_path=Path(
            os.getenv(
                "MINSAL_FALLBACK_PATH",
                str(BASE_DIR / "data" / "fallback" / "minsal_turnos.json"),
            )
        ),
        dataset_path=Path(
            os.getenv(
                "DRUG_DATASET_PATH",
                str(BASE_DIR / "data" / "raw" / "DrugData.csv"),
            )
        ),
        frontend_api_url=os.getenv(
            "API_BASE_URL",
            "http://127.0.0.1:8000",
        ).rstrip("/"),
    )
