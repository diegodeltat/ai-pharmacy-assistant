"""Recuperación semántica de fichas educativas."""

from app.config import get_settings
from app.models import RagSearchResult
from app.rag.vector_store import get_vector_store


def retrieve_medications(question: str) -> list[RagSearchResult]:
    settings = get_settings()
    results = get_vector_store().similarity_search_with_relevance_scores(
        query=question,
        k=settings.rag_top_k,
        score_threshold=settings.rag_score_threshold,
    )
    return [
        RagSearchResult(
            content=document.page_content,
            title=document.metadata.get("drug_name", "Medicamento"),
            reference=document.metadata.get("reference", "DrugData.csv"),
            score=float(score),
            metadata=document.metadata,
        )
        for document, score in results
    ]
