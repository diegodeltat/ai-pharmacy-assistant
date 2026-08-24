import asyncio
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_graph
from app.rag.ingestion import build_documents, load_dataset


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


def test_medication_route_uses_rag_tool(monkeypatch):
    async def fake_answer(_: str) -> dict:
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
