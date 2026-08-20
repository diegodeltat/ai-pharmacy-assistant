"""Clasificación determinística de intenciones y extracción de comuna."""

import csv
import re
from functools import lru_cache

from app.config import get_settings
from app.safety.input_guardrail import is_unsafe_request, normalize_for_safety


PHARMACY_WORDS = (
    "farmacia",
    "farmacias",
    "de turno",
    "abierta",
    "abiertas",
    "comuna",
)
MEDICATION_WORDS = (
    "medicamento",
    "medicamentos",
    "amoxicilina",
    "ibuprofeno",
    "paracetamol",
    "para que sirve",
    "ficha",
    "vademecum",
    "efectos secundarios",
    "efectos adversos",
    "contraindicaciones",
    "interacciones de",
    "mecanismo de accion",
    "clase farmacologica",
    "farmaco",
    "farmacos",
    "medicina",
    "dosis",
)


@lru_cache(maxsize=1)
def known_drug_names() -> set[str]:
    path = get_settings().dataset_path
    if not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return {
                normalize_for_safety(row.get("Drug Name", ""))
                for row in csv.DictReader(file)
                if row.get("Drug Name")
            }
    except (OSError, csv.Error):
        return set()


def classify_intent(question: str) -> str:
    if is_unsafe_request(question):
        return "safety"
    text = normalize_for_safety(question)
    pharmacy = any(word in text for word in PHARMACY_WORDS)
    medication = any(word in text for word in MEDICATION_WORDS) or any(
        re.search(rf"\b{re.escape(name)}\b", text)
        for name in known_drug_names()
    )
    if pharmacy and medication:
        return "mixed"
    if pharmacy:
        return "pharmacy"
    if medication:
        return "medication"
    return "general"


def extract_commune(question: str) -> str:
    match = re.search(
        r"\ben\s+([A-Za-zÁÉÍÓÚáéíóúÑñÜü\s]+?)"
        r"(?:\s+y\s+|,|\?|$)",
        question.strip(),
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""
