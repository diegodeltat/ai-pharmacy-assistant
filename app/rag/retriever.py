"""Recuperación semántica de fichas educativas."""

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import get_settings
from app.models import RagSearchResult
from app.rag.entity_resolver import normalize_drug_name
from app.rag.vector_store import get_vector_store


def retrieve_medications(
    question: str,
    expected_drug_name: str | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
    filter_by_entity: bool = True,
) -> list[RagSearchResult]:
    settings = get_settings()
    requested_k = top_k or settings.rag_top_k
    query = question
    query_filter = None
    if expected_drug_name:
        # Los documentos están en inglés. Añadir el nombre canónico mejora el
        # recall para consultas que usan la variante española. El filtro evita
        # que una similitud temática sustituya a la entidad solicitada.
        query = f"Medication: {expected_drug_name}\nUser question: {question}"
        if filter_by_entity:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.drug_name",
                        match=MatchValue(value=expected_drug_name),
                    )
                ]
            )

    results = get_vector_store().similarity_search_with_relevance_scores(
        query=query,
        k=requested_k,
        filter=query_filter,
        score_threshold=(
            settings.rag_score_threshold
            if score_threshold is None
            else score_threshold
        ),
    )

    mapped = [
        RagSearchResult(
            content=document.page_content,
            title=document.metadata.get("drug_name", "Medicamento"),
            reference=document.metadata.get("reference", "DrugData.csv"),
            score=float(score),
            metadata=document.metadata,
        )
        for document, score in results
    ]
    if expected_drug_name and filter_by_entity:
        expected = normalize_drug_name(expected_drug_name)
        mapped = [
            result
            for result in mapped
            if expected
            in {
                normalize_drug_name(result.title),
                normalize_drug_name(str(result.metadata.get("generic_name", ""))),
            }
        ]
    return mapped[:requested_k]
