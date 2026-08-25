# Asistente Farmacias IA

Asistente educativo con FastAPI, LangGraph, MINSAL en vivo y RAG semántico
sobre fichas ficticias de medicamentos. Incluye memoria por `user_id`,
guardrails, fuentes trazables y frontend Streamlit.

## Configuración local

```powershell
py -3.12 -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Completa en `.env` `OPENAI_API_KEY`, `QDRANT_URL` y `QDRANT_API_KEY`.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

## Preparar el RAG

```powershell
python -m app.rag.ingestion --dry-run
python -m app.rag.ingestion
```

Usa `--recreate` solamente para reemplazar la colección completa.
Si la colección ya existía y solo necesita los índices de metadata requeridos
por Qdrant strict mode, ejecuta:

```powershell
python -m app.rag.ingestion --indexes-only
```

## Ejecutar

```powershell
uvicorn app.main:app --reload
streamlit run frontend/app.py
```

- API: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
- Frontend: http://127.0.0.1:8501

## Pruebas y evaluación

```powershell
python -m pytest -q
python -m evaluation.evaluate
python -m evaluation.calibrate_rag
```

La evaluación end-to-end requiere que la API esté ejecutándose. La calibración
requiere OpenAI y Qdrant configurados y guarda scores, Hit@1 y abstención para
cada combinación de threshold y `top_k`.

Para comparar evaluación sin y con reranking:

```powershell
python -m evaluation.evaluate --rerank false
python -m evaluation.evaluate --rerank true
```

`--rerank` sin valor también lo activa. Los resultados se guardan por separado
en `evaluation_results_rerank_off.json` y `evaluation_results_rerank_on.json`.
El reranking usa el modelo configurado, por lo que agrega costo y latencia. Cada
resultado registra `latency_ms` y el resumen muestra p50 y p95.

## Límites

- MINSAL no informa stock, precio ni disponibilidad de medicamentos.
- El corpus RAG es ficticio y solo sirve para fines educativos.
- El asistente no diagnostica, prescribe ni recomienda medicamentos o dosis.
- La memoria conversacional utiliza un checkpointer persistente basado en SQLite, asociado a `thread_id=user_id`, permitiendo conservar el contexto entre solicitudes y reinicios del servicio.

## Despliegue

La aplicación se encuentra desplegada en Railway utilizando dos servicios independientes:

- **Backend:** FastAPI.
- **Frontend:** Streamlit.

El backend se integra con OpenAI, Qdrant y MINSAL mediante variables de entorno. La memoria conversacional utiliza SQLite con almacenamiento persistente en cloud.

El frontend consume la API del backend mediante la variable `API_BASE_URL`.

El despliegue fue validado mediante smoke tests end-to-end sobre los flujos críticos de RAG, farmacias de turno, memoria conversacional y seguridad.

## Resultados finales

- **52/52 pruebas automatizadas aprobadas** con Pytest.
- **73/73 escenarios aprobados** en la evaluación end-to-end.
- RAG validado en cloud con recuperación desde Qdrant, generación mediante OpenAI y citas verificables.
- Integración MINSAL con mecanismo de fallback explícito cuando los datos en vivo no están disponibles.
- Guardrails para consultas fuera del alcance clínico, como diagnóstico y recomendación de dosis.
- Memoria conversacional persistente.
- Frontend y backend desplegados y validados en Railway.
- Trazabilidad de las interacciones con el LLM mediante logging de operación, modelo, latencia y estado de ejecución.
