from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "rag_configured" in response.json()


def test_chat_general_request():
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


def test_chat_safety_request():
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


def test_chat_rejects_empty_question():
    response = client.post(
        "/chat",
        json={
            "user_id": "test-user",
            "pregunta": "",
        },
    )

    assert response.status_code == 422
