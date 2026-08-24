# Informe de seguridad, privacidad y calidad

## Alcance

El asistente entrega turnos de farmacias publicados por MINSAL e información
general recuperada desde un corpus educativo ficticio. No confirma stock,
precios o disponibilidad y no diagnostica, prescribe ni recomienda dosis.

## Fuentes y calidad

### MINSAL

La aplicación consulta `getLocalesTurnos.php` al responder. Valida HTTP, JSON y
campos mínimos; normaliza texto, teléfonos y horarios; descarta registros que no
corresponden a la fecha vigente. Usa timeout de 5 segundos, cache de 15 minutos
y un snapshot fechado. El snapshot siempre se presenta como respaldo, nunca
como dato vivo.

### Corpus RAG

`DrugData.csv` contiene 220 filas y 20 columnas, sin nulos. Se detectaron 94
medicamentos normalizados, 35 perfiles repetidos que afectan 81 filas y 10 NDC
reutilizados. Los NDC, precios, fabricantes y fechas de aprobación no se envían
al modelo. Las filas se agrupan por nombre y nombre genérico; los conflictos se
conservan como metadata de calidad.

El dataset declara que sus datos son ficticios. Las respuestas muestran esta
limitación y citan el medicamento y los IDs de los registros recuperados.

## Seguridad por capas

1. Pydantic valida `user_id` y longitud de la pregunta.
2. Un guardrail de entrada detecta diagnóstico, tratamiento, dosis,
   recomendaciones, roleplay e instrucciones para ignorar reglas.
3. LangGraph enruta a MINSAL, RAG, ambas herramientas o rechazo.
4. El prompt limita al LLM al contexto recuperado y exige abstención.
5. Una compuerta de entidad evita recuperar fichas para nombres desconocidos.
6. Las citas se validan y solo se muestran las fuentes realmente citadas.
7. El guardrail de salida reemplaza instrucciones de dosis accidentales.
8. La interfaz muestra fuentes, estado vivo/respaldo y advertencias.

## Privacidad

El contrato usa un UUID de sesión y no solicita nombre, RUT, correo ni historia
clínica. La memoria conversacional se gestiona mediante un checkpointer persistente basado
en SQLite. Cada conversación utiliza `thread_id=user_id`, lo que permite recuperar
el estado asociado al usuario entre solicitudes y conservarlo después de reiniciar
el servicio. No se
deben ingresar datos personales o sensibles en las preguntas.

## Evaluación

- Pruebas unitarias de normalización, horarios, dataset y routing.
- Pruebas de API para salud, validación y seguridad.
- Veinte solicitudes adversarias para el guardrail.
- Casos end-to-end en `evaluation/evaluation_cases.json`.
- Casos RAG que verifican top 1, título esperado, citas, abstención y aclaración.
- Script reproducible para comparar threshold y `top_k` con Qdrant real.
- La demo debe verificar dos turnos dependientes con el mismo `user_id`.

## Limitaciones

- El corpus no tiene autoridad clínica y está en inglés.
- La memoria no persiste entre reinicios o instancias de Render.
- La calidad y disponibilidad de MINSAL dependen de un servicio externo.
- La calibración real del umbral requiere credenciales de OpenAI/Qdrant y debe
  ejecutarse antes de la demo definitiva.