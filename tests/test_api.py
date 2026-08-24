import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    # Las versiones actuales de Starlette ejecutan el lifespan al usar el
    # cliente como context manager.
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "rag_configured" in response.json()


def test_chat_general_request(client):
    response = client.post(
        "/chat",
        json={
            "user_id": "test-user",
            "pregunta": "Hola, ¿qué puedes hacer?",
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert body["intent"] == "general"
    assert body["safety_blocked"] is False
    assert body["sources"] == []
    assert body["warnings"] == []


def test_chat_safety_request(client):
    response = client.post(
        "/chat",
        json={
            "user_id": "test-user",
            "pregunta": "¿Qué dosis de ibuprofeno debo tomar?",
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert body["intent"] == "safety"
    assert body["safety_blocked"] is True


def test_chat_accepts_per_request_rerank_toggle(client):
    response = client.post(
        "/chat",
        json={
            "user_id": "rerank-user",
            "pregunta": "Ficha de un medicamento llamado PruebaMedX",
            "rerank": True,
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert body["rerank_enabled"] is True
    assert body["rerank_applied"] is False


def test_chat_rejects_empty_question(client):
    response = client.post(
        "/chat",
        json={
            "user_id": "test-user",
            "pregunta": "",
        },
    )

    assert response.status_code == 422
