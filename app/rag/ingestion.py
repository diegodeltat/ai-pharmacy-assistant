"""Indexación offline del dataset educativo de medicamentos."""

import argparse
import csv
import re
import uuid
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from app.config import get_settings
from app.rag.vector_store import build_embedding_model


REQUIRED_COLUMNS = {
    "Drug ID",
    "Drug Name",
    "Generic Name",
    "Drug Class",
    "Indications",
    "Dosage Form",
    "Strength",
    "Route of Administration",
    "Mechanism of Action",
    "Side Effects",
    "Contraindications",
    "Interactions",
    "Warnings and Precautions",
    "Pregnancy Category",
    "Storage Conditions",
    "Manufacturer",
    "Approval Date",
    "Availability",
    "NDC",
    "Price",
}

EDUCATIONAL_NOTICE = (
    "This record comes from a fictional educational dataset and must not be "
    "used for diagnosis, prescribing, dosage, or clinical decisions."
)


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _unique_values(rows: list[dict[str, str]], column: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = normalize_text(row.get(column))
        key = value.casefold()
        if value and key not in seen:
            values.append(value)
            seen.add(key)
    return values


def load_dataset(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        actual_columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - actual_columns
        if missing:
            raise ValueError(
                "El dataset no contiene las columnas requeridas: "
                f"{sorted(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError("El dataset no contiene registros.")

    empty_required = [
        index
        for index, row in enumerate(rows, start=2)
        if not normalize_text(row.get("Drug ID"))
        or not normalize_text(row.get("Drug Name"))
        or not normalize_text(row.get("Generic Name"))
    ]
    if empty_required:
        raise ValueError(
            "Hay filas sin identificador o nombre de medicamento: "
            f"{empty_required[:10]}"
        )
    return rows


def build_documents(rows: list[dict[str, str]]) -> list[Document]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            normalize_text(row["Drug Name"]).casefold(),
            normalize_text(row["Generic Name"]).casefold(),
        )
        grouped[key].append(row)

    documents: list[Document] = []
    aggregate_fields = [
        ("Drug class", "Drug Class"),
        ("General indications", "Indications"),
        ("Mechanism of action", "Mechanism of Action"),
        ("Reported side effects", "Side Effects"),
        ("Contraindications", "Contraindications"),
        ("Interactions", "Interactions"),
        ("Warnings and precautions", "Warnings and Precautions"),
    ]

    for grouped_rows in grouped.values():
        drug_name = normalize_text(grouped_rows[0]["Drug Name"])
        generic_name = normalize_text(grouped_rows[0]["Generic Name"])
        drug_ids = _unique_values(grouped_rows, "Drug ID")
        lines = [
            f"Medication: {drug_name}",
            f"Generic name: {generic_name}",
        ]
        conflicts: list[str] = []

        for label, column in aggregate_fields:
            values = _unique_values(grouped_rows, column)
            if len(values) > 1:
                conflicts.append(column)
            if values:
                lines.append(f"{label}: {'; '.join(values)}")

        presentations: list[str] = []
        for row in grouped_rows:
            presentation = " | ".join(
                filter(
                    None,
                    [
                        normalize_text(row.get("Dosage Form")),
                        normalize_text(row.get("Strength")),
                        normalize_text(row.get("Route of Administration")),
                        normalize_text(row.get("Availability")),
                    ],
                )
            )
            if presentation and presentation not in presentations:
                presentations.append(presentation)
        if presentations:
            lines.append(f"Available presentations: {'; '.join(presentations)}")

        lines.append(f"Educational dataset notice: {EDUCATIONAL_NOTICE}")
        content = "\n".join(lines)
        reference = f"DrugData.csv — registros {', '.join(drug_ids)}"

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "DrugData.csv",
                    "drug_name": drug_name,
                    "generic_name": generic_name,
                    "drug_ids": ",".join(drug_ids),
                    "reference": reference,
                    "original_language": "en",
                    "fictional_source": True,
                    "quality_conflict": bool(conflicts),
                    "conflicting_fields": ",".join(conflicts),
                },
            )
        )

    return sorted(
        documents,
        key=lambda document: document.metadata["drug_name"].casefold(),
    )


def generate_document_ids(documents: list[Document]) -> list[str]:
    return [
        str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "|".join(
                    [
                        document.metadata["drug_name"].casefold(),
                        document.metadata["generic_name"].casefold(),
                    ]
                ),
            )
        )
        for document in documents
    ]


def ensure_payload_indexes(
    client: QdrantClient,
    collection_name: str,
) -> None:
    """Crea índices requeridos por filtros en clusters con strict mode."""

    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.drug_name",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )


def index_payloads_only() -> None:
    """Repara índices de payload sin recalcular embeddings ni subir documentos."""

    settings = get_settings()
    if not settings.rag_configured:
        raise RuntimeError(
            "Faltan OPENAI_API_KEY, QDRANT_URL o QDRANT_API_KEY."
        )

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )
    if not client.collection_exists(settings.qdrant_collection):
        raise RuntimeError(
            f"La colección {settings.qdrant_collection!r} no existe."
        )
    ensure_payload_indexes(client, settings.qdrant_collection)


def index_documents(documents: list[Document], recreate: bool = False) -> int:
    settings = get_settings()
    if not settings.rag_configured:
        raise RuntimeError(
            "Faltan OPENAI_API_KEY, QDRANT_URL o QDRANT_API_KEY."
        )

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )
    exists = client.collection_exists(settings.qdrant_collection)
    if recreate and exists:
        client.delete_collection(settings.qdrant_collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )
    else:
        collection = client.get_collection(settings.qdrant_collection)
        vectors = collection.config.params.vectors
        vector_size = getattr(vectors, "size", None)
        if vector_size != settings.embedding_dimensions:
            raise RuntimeError(
                "La colección existente usa una dimensionalidad incompatible: "
                f"{vector_size}; se esperaban {settings.embedding_dimensions}. "
                "Usa --recreate para reemplazarla conscientemente."
            )

    ensure_payload_indexes(client, settings.qdrant_collection)

    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=build_embedding_model(),
    )
    store.add_documents(documents, ids=generate_document_ids(documents))
    stored_points = client.get_collection(settings.qdrant_collection).points_count
    if stored_points is None or stored_points < len(documents):
        raise RuntimeError(
            "Qdrant no confirmó la cantidad esperada de documentos indexados."
        )
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa DrugData.csv en Qdrant")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--indexes-only",
        action="store_true",
        help="Crea índices de payload sin recalcular embeddings.",
    )
    args = parser.parse_args()

    if args.indexes_only:
        index_payloads_only()
        print("Índices de payload verificados.")
        return

    settings = get_settings()
    rows = load_dataset(settings.dataset_path)
    documents = build_documents(rows)
    print(f"Filas leídas: {len(rows)}")
    print(f"Documentos generados: {len(documents)}")
    print(
        "Documentos con conflictos: "
        f"{sum(bool(item.metadata['quality_conflict']) for item in documents)}"
    )
    if not args.dry_run:
        stored = index_documents(documents, recreate=args.recreate)
        print(f"Documentos indexados: {stored}")


if __name__ == "__main__":
    main()
