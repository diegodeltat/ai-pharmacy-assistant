"""Interfaz conversacional Streamlit que consume la API FastAPI."""

import os
import uuid

import httpx
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Asistente Farmacias IA",
    page_icon="💊",
    layout="centered",
)
st.title("💊 Asistente Farmacias IA")
st.caption(
    "Consulta farmacias de turno y fichas educativas. No entrega diagnósticos, "
    "tratamientos ni recomendaciones de dosis."
)

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


def reset_conversation() -> None:
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.messages = []


with st.sidebar:
    st.subheader("Sesión")
    st.code(st.session_state.user_id, language=None)
    st.button("Nueva conversación", on_click=reset_conversation)
    st.divider()
    st.markdown(
        "**Ejemplos**\n\n"
        "- ¿Qué farmacias están de turno en Ñuñoa?\n"
        "- ¿Para qué sirve la amoxicilina?\n"
        "- ¿Qué dosis debo tomar? (prueba de seguridad)"
    )


def render_assistant_message(message: dict) -> None:
    st.markdown(message["content"])
    for warning in message.get("warnings", []):
        st.warning(warning)
    sources = message.get("sources", [])
    if sources:
        with st.expander("Fuentes utilizadas"):
            for source in sources:
                status = ""
                if source.get("source_type") == "minsal":
                    status = " · en vivo" if source.get("live_data") else " · respaldo"
                score = source.get("score")
                score_text = f" · score {score:.3f}" if score is not None else ""
                st.markdown(
                    f"**{source['title']}**{status}{score_text}\n\n"
                    f"{source['reference']}"
                )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message["content"])


if question := st.chat_input("Escribe tu consulta"):
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Consultando fuentes..."):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/chat",
                    json={
                        "user_id": st.session_state.user_id,
                        "pregunta": question,
                    },
                    timeout=35,
                )
                response.raise_for_status()
                body = response.json()
                assistant_message = {
                    "role": "assistant",
                    "content": body["respuesta"],
                    "sources": body.get("sources", []),
                    "warnings": body.get("warnings", []),
                }
                render_assistant_message(assistant_message)
                st.session_state.messages.append(assistant_message)
            except (httpx.RequestError, httpx.HTTPStatusError, ValueError):
                error_message = {
                    "role": "assistant",
                    "content": (
                        "No fue posible comunicarse con el asistente. "
                        "Verifica que la API esté disponible e intenta nuevamente."
                    ),
                    "sources": [],
                    "warnings": [],
                }
                st.error(error_message["content"])
                st.session_state.messages.append(error_message)
