# Documento de transferencia

## Estado actual

La implementación local incluye:

- FastAPI con `/health` y `POST /chat`.
- LangGraph con rutas `pharmacy`, `medication`, `mixed`, `safety` y `general`.
- Memoria conversacional mediante `InMemorySaver` y `thread_id=user_id`.
- Tool MINSAL asíncrona con validación, cache, vigencia y fallback fechado.
- Pipeline offline del dataset Kaggle: 220 filas transformadas en 94 documentos.
- OpenAI Embeddings y Qdrant con inicialización diferida.
- Respuestas RAG en español con fuentes y advertencia de corpus ficticio.
- Resolución de nombres genéricos y alias españoles antes de consultar Qdrant.
- Validación de citas y exposición exclusiva de fuentes citadas.
- Reranking LLM opcional mediante request o CLI de evaluación.
- Guardrails de entrada y salida.
- Frontend Streamlit.
- Dockerfile y blueprint para dos servicios en Render.
- 52 pruebas automatizadas.

## Comandos

```powershell
.\.venv-win\Scripts\Activate.ps1
python -m app.rag.ingestion --dry-run
python -m app.rag.ingestion
python -m pytest -q
uvicorn app.main:app --reload
streamlit run frontend/app.py
```

## Configuración externa pendiente

Para habilitar e indexar el RAG deben definirse:

- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

La colección esperada es `drug_information_v1`, con vectores de 256
dimensiones. La indexación no se ejecuta durante una request.

## Validación realizada

- Dataset: 220 filas, 94 documentos y 11 medicamentos con conflictos internos.
- Pruebas: 52 aprobadas.
- API local: salud, general y rechazo de dosis verificados.
- MINSAL: consulta end-to-end verificada con `live_data=true`.
- Streamlit: servidor iniciado y respuesta HTTP 200.

No se ejecutó la indexación real en Qdrant ni una consulta real al LLM porque
el workspace no contiene credenciales. Tampoco se realizó el despliegue en
Render; requiere las cuentas, secretos y URLs definitivas.

## Límites

- El corpus de medicamentos es ficticio y exclusivamente educativo.
- El asistente no recomienda medicamentos, dosis ni tratamientos.
- MINSAL no confirma stock, precio ni disponibilidad.
- La memoria actual se pierde al reiniciar el proceso.
