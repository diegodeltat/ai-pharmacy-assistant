# Evaluación y criterios de cierre del RAG

## Decisiones de diseño

El RAG conserva OpenAI Embeddings, Qdrant y el corpus agrupado por medicamento.
Antes de ejecutar búsqueda semántica se resuelve una entidad contra `Drug Name`,
`Generic Name` y un conjunto controlado de alias en español. Esta compuerta es
necesaria porque los scores observados en consultas válidas y nombres inventados
se solapan; un threshold por sí solo no permite una abstención confiable.

Si no se reconoce un medicamento, el sistema solicita el nombre exacto y no
consulta Qdrant. Cuando reconoce uno, la búsqueda se enriquece con su nombre
canónico y solo conserva resultados cuya metadata corresponde a esa entidad.

Las citas se validan después de la generación. Cada `[Fuente N]` debe apuntar a
un documento existente. Se hace un reintento si faltan citas; si continúa el
problema, se devuelve una abstención controlada. La API expone únicamente las
fuentes realmente citadas y renumera las etiquetas para mantener consistencia.

## Casos evaluados

Los casos `rag`, `rag_no_answer` y `rag_unsupported` de
`evaluation/evaluation_cases.json` validan:

- intención `medication`;
- fuente esperada y resultado top 1;
- presencia y rango válido de citas;
- abstención sin fuentes para nombres inexistentes;
- abstención para precio, fabricante y aprobación, campos excluidos del contexto;
- solicitud de aclaración cuando falta una entidad verificable.

El caso de azitromicina se clasifica como no respondible porque ese medicamento
no existe en `DrugData.csv`.

## Calibración

Con Qdrant y OpenAI configurados:

```powershell
python -m evaluation.calibrate_rag
```

El comando compara thresholds `0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.75` y
valores de `top_k` de `1, 2, 4`. Registra títulos, scores, Hit@1 y abstención de
negativos en `evaluation/rag_calibration_results.json`.

No se debe cambiar `RAG_SCORE_THRESHOLD` basándose en una sola consulta. La
configuración elegida debe maximizar primero la abstención de negativos y luego
Hit@1, sin degradar los casos respondibles.

## Comparación con reranking

El reranking se puede activar por request sin reiniciar el backend. Recupera un
conjunto más amplio, usa el modelo para ordenarlo por relevancia y vuelve a
aplicar la compuerta de identidad antes de responder. Esto impide que el
reranker introduzca un medicamento distinto del solicitado.

```powershell
python -m evaluation.evaluate --rerank false
python -m evaluation.evaluate --rerank true
```

También se acepta `--rerank` como atajo para `--rerank true`. Las ejecuciones
guardan archivos separados. Deben compararse calidad, latencia y costo antes de
activar `RAG_RERANK_ENABLED=true` como valor predeterminado. Los identificadores
de conversación se separan por modo para evitar contaminación de memoria, y el
reporte incluye latencia por caso, p50 y p95.

El filtro por medicamento requiere un índice `keyword` en
`metadata.drug_name`. La ingesta lo crea automáticamente. Para reparar una
colección creada antes de este cambio sin regenerar embeddings:

```powershell
python -m app.rag.ingestion --indexes-only
```

## Criterio de cierre

- 100% de casos respondibles recuperan el medicamento esperado en top 1.
- 100% de respuestas factuales contienen citas válidas.
- Nombres inexistentes y consultas sin nombre no muestran fuentes irrelevantes.
- Las pruebas unitarias y la evaluación end-to-end terminan sin fallos.
