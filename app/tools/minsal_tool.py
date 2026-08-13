"""Cliente resiliente para farmacias de turno publicadas por MINSAL."""

import json
import re
import time
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.config import get_settings


MINSAL_TURNOS_URL = (
    "https://midas.minsal.cl/farmacia_v2/WS/getLocalesTurnos.php"
)
MAX_RESULTS = 5
REQUIRED_FIELDS = {
    "local_nombre",
    "comuna_nombre",
    "local_direccion",
    "funcionamiento_hora_apertura",
    "funcionamiento_hora_cierre",
    "fecha",
}

_cache_records: list[dict[str, Any]] | None = None
_cache_at = 0.0
_http_client: httpx.AsyncClient | None = None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def normalize_for_comparison(value: str) -> str:
    normalized = unicodedata.normalize("NFD", normalize_text(value).casefold())
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )


def format_time(value: Any) -> str:
    match = re.match(r"^(\d{2}:\d{2})", normalize_text(value))
    return match.group(1) if match else "Horario no informado"


def format_schedule(opening_time: Any, closing_time: Any) -> str:
    opening = format_time(opening_time)
    closing = format_time(closing_time)
    if "no informado" in opening.casefold() or "no informado" in closing.casefold():
        return "Horario no informado"
    if closing < opening:
        return f"{opening} a {closing} del día siguiente"
    return f"{opening} a {closing}"


def normalize_phone(value: Any) -> str:
    phone = normalize_text(value)
    if len(re.sub(r"\D", "", phone)) < 7:
        return "No informado"
    return phone


def normalize_pharmacy(record: dict[str, Any]) -> dict[str, str]:
    return {
        "nombre": normalize_text(record.get("local_nombre"))
        or "Nombre no informado",
        "comuna": normalize_text(record.get("comuna_nombre"))
        or "Comuna no informada",
        "direccion": normalize_text(record.get("local_direccion"))
        or "Dirección no informada",
        "horario": format_schedule(
            record.get("funcionamiento_hora_apertura"),
            record.get("funcionamiento_hora_cierre"),
        ),
        "telefono": normalize_phone(record.get("local_telefono")),
        "fecha": normalize_text(record.get("fecha")) or "Fecha no informada",
    }


def _parse_date(value: Any) -> date | None:
    text = normalize_text(value)
    for pattern in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    return None


def _validate_records(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("MINSAL entregó un JSON con formato inesperado.")
    valid = [
        record
        for record in data
        if isinstance(record, dict) and REQUIRED_FIELDS.issubset(record)
    ]
    if not valid:
        raise ValueError("MINSAL no entregó registros con el esquema esperado.")
    return valid


def _load_fallback(path: Path) -> tuple[list[dict[str, Any]], str] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = _validate_records(payload.get("records"))
        captured_at = normalize_text(payload.get("captured_at")) or "desconocida"
        return records, captured_at
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        settings = get_settings()
        _http_client = httpx.AsyncClient(
            timeout=settings.minsal_timeout_seconds,
            follow_redirects=True,
            headers={
                "Accept": "application/json",
                "User-Agent": "AsistenteFarmaciasIA/1.0 (proyecto educativo)",
            },
        )
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None


async def fetch_turn_pharmacies() -> list[dict[str, Any]]:
    global _cache_at, _cache_records
    settings = get_settings()
    now = time.monotonic()
    if (
        _cache_records is not None
        and now - _cache_at < settings.minsal_cache_seconds
    ):
        return _cache_records

    response = await _get_client().get(MINSAL_TURNOS_URL)
    response.raise_for_status()
    try:
        records = _validate_records(response.json())
    except json.JSONDecodeError as error:
        raise ValueError("MINSAL entregó JSON inválido.") from error
    _cache_records = records
    _cache_at = now
    return records


async def _get_source_records() -> dict[str, Any]:
    settings = get_settings()
    try:
        records = await fetch_turn_pharmacies()
        today = datetime.now(ZoneInfo("America/Santiago")).date()
        current_records = [
            record for record in records if _parse_date(record.get("fecha")) == today
        ]
        if current_records:
            return {
                "records": current_records,
                "live_data": True,
                "captured_at": today.isoformat(),
                "error": None,
            }
        live_error = "MINSAL no entregó registros vigentes para la fecha actual."
    except httpx.TimeoutException:
        live_error = "La consulta a MINSAL excedió el tiempo de espera."
    except httpx.HTTPStatusError as error:
        live_error = f"MINSAL respondió HTTP {error.response.status_code}."
    except (httpx.RequestError, ValueError) as error:
        live_error = f"No fue posible validar la respuesta de MINSAL: {error}"

    fallback = _load_fallback(settings.minsal_fallback_path)
    if fallback:
        records, captured_at = fallback
        return {
            "records": records,
            "live_data": False,
            "captured_at": captured_at,
            "error": live_error,
        }
    return {
        "records": [],
        "live_data": False,
        "captured_at": None,
        "error": live_error,
    }


async def find_pharmacies_by_commune(
    commune: str,
    max_results: int = MAX_RESULTS,
) -> dict[str, Any]:
    normalized_commune = normalize_for_comparison(commune)
    if not normalized_commune:
        return {
            "success": False,
            "commune": commune,
            "pharmacies": [],
            "message": "Debes indicar una comuna para realizar la búsqueda.",
            "source": "MINSAL",
            "live_data": False,
            "captured_at": None,
        }

    source = await _get_source_records()
    if not source["records"]:
        return {
            "success": False,
            "commune": commune,
            "pharmacies": [],
            "message": (
                f"{source['error']} No existe un fallback local disponible."
            ),
            "source": "MINSAL",
            "live_data": False,
            "captured_at": None,
        }

    matching = [
        normalize_pharmacy(record)
        for record in source["records"]
        if normalize_for_comparison(record.get("comuna_nombre", ""))
        == normalized_commune
    ]
    if not matching:
        data_label = "actual" if source["live_data"] else "de respaldo"
        return {
            "success": True,
            "commune": commune,
            "pharmacies": [],
            "message": (
                f"No encontré farmacias de turno para {commune} en los "
                f"datos {data_label} de MINSAL."
            ),
            "source": "MINSAL",
            "live_data": source["live_data"],
            "captured_at": source["captured_at"],
        }

    warning = ""
    if not source["live_data"]:
        warning = (
            f" Datos de respaldo capturados el {source['captured_at']}; "
            "no representan el estado actual."
        )
    return {
        "success": True,
        "commune": commune,
        "pharmacies": matching[:max_results],
        "message": f"Se encontraron {len(matching)} farmacias.{warning}",
        "source": "MINSAL",
        "live_data": source["live_data"],
        "captured_at": source["captured_at"],
        "source_error": source["error"],
    }
