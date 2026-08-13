"""Conexión diferida a OpenAI Embeddings y Qdrant."""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import get_settings


def ensure_rag_configuration() -> None:
    if not get_settings().rag_configured:
        raise RuntimeError(
            "El RAG no está configurado. Define OPENAI_API_KEY, "
            "QDRANT_URL y QDRANT_API_KEY."
        )


@lru_cache(maxsize=1)
def build_embedding_model() -> OpenAIEmbeddings:
    ensure_rag_configuration()
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        api_key=settings.openai_api_key,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    ensure_rag_configuration()
    settings = get_settings()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=30,
    )
    if not client.collection_exists(settings.qdrant_collection):
        raise RuntimeError(
            f"La colección {settings.qdrant_collection!r} no está indexada."
        )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=build_embedding_model(),
    )
