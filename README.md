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

## Preparar el RAG

```powershell
python -m app.rag.ingestion --dry-run
python -m app.rag.ingestion
```

Usa `--recreate` solamente para reemplazar la colección completa.

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
```

La evaluación end-to-end requiere que la API esté ejecutándose.

## Límites

- MINSAL no informa stock, precio ni disponibilidad de medicamentos.
- El corpus RAG es ficticio y solo sirve para fines educativos.
- El asistente no diagnostica, prescribe ni recomienda medicamentos o dosis.
- La memoria actual vive en el proceso y se pierde al reiniciar.
