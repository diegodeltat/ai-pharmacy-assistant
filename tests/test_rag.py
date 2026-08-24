import asyncio
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.documents import Document

from app.agent.graph import build_graph
from app.agent.router import classify_intent
from app.models import RagSearchResult
from app.rag.entity_resolver import resolve_medication_entity
from app.rag.ingestion import build_documents, ensure_payload_indexes, load_dataset
from app.rag.retriever import retrieve_medications
from app.tools.rag_tool import (
    RAG_CITATION_WARNING,
    answer_medication_question,
    remap_citations,
    parse_rerank_order,
    rerank_results,
    validate_citations,
)


graph = build_graph(InMemorySaver())

def invoke_graph(question: str, user_id: str = "test-rag") -> dict:
    return asyncio.run(
        graph.ainvoke(
            {
                "user_id": user_id,
                "question": question,
                "safety_blocked": False,
            },
            config={"configurable": {"thread_id": user_id}},
        )
    )


def test_dataset_builds_grouped_documents():
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "DrugData.csv"
    rows = load_dataset(path)
    documents = build_documents(rows)

    assert len(rows) == 220
    assert len(documents) == 94
    amoxicillin = next(
        document
        for document in documents
        if document.metadata["drug_name"] == "Amoxicillin"
    )
    assert "Bacterial Infections" in amoxicillin.page_content
    assert amoxicillin.metadata["drug_ids"] == "2,147,172,197,206"
    assert "Price" not in amoxicillin.page_content
    assert "NDC" not in amoxicillin.page_content


def test_ingestion_creates_drug_name_payload_index():
    captured = {}

    class FakeClient:
        def create_payload_index(self, **kwargs):
            captured.update(kwargs)

    ensure_payload_indexes(FakeClient(), "drug_information_v1")

    assert captured["collection_name"] == "drug_information_v1"
    assert captured["field_name"] == "metadata.drug_name"
    assert captured["field_schema"].value == "keyword"
    assert captured["wait"] is True


def test_medication_route_uses_rag_tool(monkeypatch):
    async def fake_answer(*args, **kwargs) -> dict:
        return {
            "answer": "Información recuperada [Fuente 1]",
            "sources": [{"source_type": "rag", "title": "Amoxicillin"}],
            "warnings": ["Corpus educativo ficticio"],
            "safety_blocked": False,
        }

    monkeypatch.setattr("app.agent.nodes.answer_medication_question", fake_answer)
    result = invoke_graph("¿Para qué sirve la amoxicilina?")

    assert result["intent"] == "medication"
    assert "Fuente 1" in result["response"]
    assert result["sources"][0]["title"] == "Amoxicillin"


def test_spanish_aliases_route_and_resolve_to_corpus_names():
    assert resolve_medication_entity("¿Qué información hay sobre omeprazol?") == (
        "Omeprazole"
    )
    assert resolve_medication_entity("Resume la ficha de amlodipino") == (
        "Amlodipine"
    )
    assert resolve_medication_entity("Ficha del ácido acetilsalicílico") == (
        "Aspirin"
    )
    assert classify_intent("Información general sobre omeprazol") == "medication"
    assert classify_intent("Información general sobre amlodipino") == "medication"


def test_unknown_medication_abstains_without_retrieval(monkeypatch):
    def unexpected_retrieval(*args, **kwargs):
        raise AssertionError("No se debe consultar Qdrant sin una entidad válida")

    monkeypatch.setattr(
        "app.tools.rag_tool.retrieve_medications",
        unexpected_retrieval,
    )
    result = asyncio.run(
        answer_medication_question(
            "¿Qué información tienes sobre un medicamento llamado PruebaMedX?"
        )
    )

    assert result["sources"] == []
    assert result["clarification_required"] is True
    assert "Indica el nombre exacto" in result["answer"]


def test_excluded_attribute_abstains_without_retrieval(monkeypatch):
    def unexpected_retrieval(*args, **kwargs):
        raise AssertionError("No se debe recuperar un campo excluido del corpus")

    monkeypatch.setattr(
        "app.tools.rag_tool.retrieve_medications",
        unexpected_retrieval,
    )
    result = asyncio.run(
        answer_medication_question("¿Cuál es el precio de la amoxicilina?")
    )

    assert result["sources"] == []
    assert "precio" in result["answer"]
    assert "No se encontró información suficiente" in result["answer"]


def test_retrieval_filters_qdrant_by_resolved_entity(monkeypatch):
    captured = {}

    class FakeStore:
        def similarity_search_with_relevance_scores(self, **kwargs):
            captured.update(kwargs)
            return [
                (
                    Document(
                        page_content="Medication: Omeprazole",
                        metadata={
                            "drug_name": "Omeprazole",
                            "generic_name": "Omeprazole",
                            "reference": "registro 5",
                        },
                    ),
                    0.9,
                )
            ]

    monkeypatch.setattr("app.rag.retriever.get_vector_store", lambda: FakeStore())
    results = retrieve_medications(
        "Información sobre omeprazol",
        expected_drug_name="Omeprazole",
    )

    assert results[0].title == "Omeprazole"
    assert "Omeprazole" in captured["query"]
    assert captured["filter"].must[0].key == "metadata.drug_name"
    assert captured["filter"].must[0].match.value == "Omeprazole"


def test_citation_validation_and_remapping():
    valid, indices = validate_citations(
        "Dato [Fuente 2]. Otro dato [Fuente 1].",
        2,
    )
    assert valid is True
    assert indices == [2, 1]
    assert remap_citations("A [Fuente 2]. B [Fuente 1].", indices) == (
        "A [Fuente 1]. B [Fuente 2]."
    )

    invalid, _ = validate_citations("Dato [Fuente 3]", 2)
    assert invalid is False


def test_rerank_parser_and_llm_order(monkeypatch):
    results = [
        RagSearchResult(
            content="Documento uno",
            title="Uno",
            reference="1",
            score=0.8,
        ),
        RagSearchResult(
            content="Documento dos",
            title="Dos",
            reference="2",
            score=0.7,
        ),
    ]

    assert parse_rerank_order('{"ranking": [2]}', 2) == [2, 1]

    class FakeResponse:
        content = '```json\n{"ranking": [2, 1]}\n```'

    class FakeLlm:
        async def ainvoke(self, messages):
            return FakeResponse()

    monkeypatch.setattr("app.tools.rag_tool.get_llm", lambda: FakeLlm())
    ranked, applied = asyncio.run(rerank_results("pregunta", results))

    assert applied is True
    assert [result.title for result in ranked] == ["Dos", "Uno"]


def test_answer_returns_only_cited_sources(monkeypatch):
    results = [
        RagSearchResult(
            content="Medication: Amoxicillin",
            title="Amoxicillin",
            reference="registro 1",
            score=0.9,
        ),
        RagSearchResult(
            content="Medication: Amoxicillin",
            title="Amoxicillin presentación 2",
            reference="registro 2",
            score=0.8,
        ),
    ]

    def fake_retrieve(*args, **kwargs):
        return results

    async def fake_generate(*args, **kwargs):
        return "Información sustentada [Fuente 2]."

    monkeypatch.setattr("app.tools.rag_tool.retrieve_medications", fake_retrieve)
    monkeypatch.setattr("app.tools.rag_tool._generate_answer", fake_generate)

    result = asyncio.run(answer_medication_question("Ficha de amoxicilina"))

    assert result["answer"] == "Información sustentada [Fuente 1]."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["reference"] == "registro 2"


def test_answer_can_enable_reranking_per_request(monkeypatch):
    captured = {}
    exact = RagSearchResult(
        content="Medication: Amoxicillin",
        title="Amoxicillin",
        reference="registro exacto",
        score=0.8,
        metadata={"generic_name": "Amoxicillin"},
    )
    irrelevant = RagSearchResult(
        content="Medication: Loratadine",
        title="Loratadine",
        reference="registro irrelevante",
        score=0.9,
        metadata={"generic_name": "Loratadine"},
    )

    def fake_retrieve(*args, **kwargs):
        captured.update(kwargs)
        return [irrelevant, exact]

    async def fake_rerank(question, results):
        return [exact, irrelevant], True

    async def fake_generate(*args, **kwargs):
        return "Información sustentada [Fuente 1]."

    monkeypatch.setattr("app.tools.rag_tool.retrieve_medications", fake_retrieve)
    monkeypatch.setattr("app.tools.rag_tool.rerank_results", fake_rerank)
    monkeypatch.setattr("app.tools.rag_tool._generate_answer", fake_generate)

    result = asyncio.run(
        answer_medication_question(
            "Ficha de amoxicilina",
            rerank_enabled=True,
        )
    )

    assert captured["filter_by_entity"] is False
    assert captured["top_k"] >= 10
    assert result["rerank_applied"] is True
    assert [source["title"] for source in result["sources"]] == ["Amoxicillin"]


def test_missing_citations_produces_controlled_abstention(monkeypatch):
    result = RagSearchResult(
        content="Medication: Amoxicillin",
        title="Amoxicillin",
        reference="registro 1",
        score=0.9,
    )

    monkeypatch.setattr(
        "app.tools.rag_tool.retrieve_medications",
        lambda *args, **kwargs: [result],
    )

    async def fake_generate(*args, **kwargs):
        return "Información sin cita."

    monkeypatch.setattr("app.tools.rag_tool._generate_answer", fake_generate)
    response = asyncio.run(answer_medication_question("Ficha de amoxicilina"))

    assert response["sources"] == []
    assert RAG_CITATION_WARNING in response["warnings"]
    assert "suficientemente fundamentada" in response["answer"]


def test_safety_route_blocks_dosage():
    result = invoke_graph("¿Qué dosis de ibuprofeno debo tomar?", "safe-user")
    assert result["intent"] == "safety"
    assert result["safety_blocked"] is True


def test_general_route():
    result = invoke_graph("Hola, ¿qué puedes hacer?", "general-user")
    assert result["intent"] == "general"
    assert result["safety_blocked"] is False


def test_pharmacy_followup_uses_memory(monkeypatch):
    async def fake_pharmacies(commune: str) -> dict:
        return {
            "success": True,
            "commune": commune,
            "pharmacies": [],
            "message": f"Sin resultados para {commune}",
            "source": "MINSAL",
            "live_data": True,
            "captured_at": "2026-08-12",
        }

    monkeypatch.setattr("app.agent.nodes.find_pharmacies_by_commune", fake_pharmacies)
    user_id = "memory-pharmacy-user"
    first = invoke_graph("¿Farmacias de turno en Ñuñoa?", user_id)
    second = invoke_graph("¿Y en Providencia?", user_id)

    assert first["intent"] == "pharmacy"
    assert second["intent"] == "pharmacy"
    assert second["last_commune"] == "Providencia"
