"""Resolución conservadora de nombres de medicamentos del corpus."""

import csv
import re
from functools import lru_cache

from app.config import get_settings
from app.safety.input_guardrail import normalize_for_safety


# Alias de traducción controlados. Solo se habilitan si el medicamento canónico
# existe realmente en el dataset configurado.
SPANISH_ALIASES = {
    "acetaminofen": "Paracetamol",
    "acido acetilsalicilico": "Aspirin",
    "amoxicilina": "Amoxicillin",
    "amlodipino": "Amlodipine",
    "aspirina": "Aspirin",
    "atorvastatina": "Atorvastatin",
    "cetirizina": "Cetirizine",
    "ibuprofeno": "Ibuprofen",
    "loratadina": "Loratadine",
    "metformina": "Metformin",
    "omeprazol": "Omeprazole",
    "rosuvastatina": "Rosuvastatin",
    "simvastatina": "Simvastatin",
    "valaciclovir": "Valacyclovir",
}

def normalize_drug_name(value: str) -> str:
    """Normaliza un nombre para compararlo sin tildes ni puntuación."""

    normalized = normalize_for_safety(value)
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


@lru_cache(maxsize=1)
def drug_aliases() -> dict[str, str]:
    """Devuelve alias normalizado -> nombre canónico existente en el corpus."""

    path = get_settings().dataset_path
    if not path.exists():
        return {}

    aliases: dict[str, str] = {}
    canonical_names: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                drug_name = (row.get("Drug Name") or "").strip()
                generic_name = (row.get("Generic Name") or "").strip()
                if not drug_name:
                    continue

                canonical_names[normalize_drug_name(drug_name)] = drug_name
                for candidate in (drug_name, generic_name):
                    alias = normalize_drug_name(candidate)
                    if alias:
                        aliases.setdefault(alias, drug_name)
    except (OSError, csv.Error):
        return {}

    for alias, requested_canonical in SPANISH_ALIASES.items():
        canonical = canonical_names.get(normalize_drug_name(requested_canonical))
        if canonical:
            aliases[normalize_drug_name(alias)] = canonical

    return aliases


def resolve_medication_entity(question: str) -> str | None:
    """Encuentra el nombre canónico mencionado explícitamente en la pregunta."""

    text = f" {normalize_drug_name(question)} "
    matches = [
        (alias, canonical)
        for alias, canonical in drug_aliases().items()
        if f" {alias} " in text
    ]
    if not matches:
        return None

    # Los nombres combinados deben ganar sobre coincidencias parciales.
    return max(matches, key=lambda item: len(item[0]))[1]
