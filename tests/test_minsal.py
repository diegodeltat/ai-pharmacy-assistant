import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from app.tools import minsal_tool
from app.tools.minsal_tool import (
    find_pharmacies_by_commune,
    format_schedule,
    normalize_for_comparison,
    normalize_phone,
    normalize_text,
)


def test_normalization_and_schedule():
    assert normalize_text("  FARMACIA   CENTRAL  ") == "FARMACIA CENTRAL"
    assert normalize_for_comparison("Ñuñoa") == "nunoa"
    assert normalize_phone("+560") == "No informado"
    assert normalize_phone("+56223456789") == "+56223456789"
    assert format_schedule("09:00:00", "21:00:00") == "09:00 a 21:00"
    assert (
        format_schedule("09:00:00", "08:59:00")
        == "09:00 a 08:59 del día siguiente"
    )


def test_live_pharmacy_response(monkeypatch):
    today = datetime.now(ZoneInfo("America/Santiago")).date().isoformat()

    async def fake_fetch():
        return [
            {
                "local_nombre": "FARMACIA CENTRAL",
                "comuna_nombre": "ÑUÑOA",
                "local_direccion": "IRARRÁZAVAL 100",
                "funcionamiento_hora_apertura": "09:00:00",
                "funcionamiento_hora_cierre": "08:59:00",
                "local_telefono": "+56223456789",
                "fecha": today,
            }
        ]

    monkeypatch.setattr(minsal_tool, "fetch_turn_pharmacies", fake_fetch)
    result = asyncio.run(find_pharmacies_by_commune("Ñuñoa"))
    assert result["success"] is True
    assert result["live_data"] is True
    assert result["pharmacies"][0]["nombre"] == "FARMACIA CENTRAL"


def test_timeout_uses_dated_fallback(monkeypatch):
    async def timeout():
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(minsal_tool, "fetch_turn_pharmacies", timeout)
    source = asyncio.run(minsal_tool._get_source_records())
    assert source["live_data"] is False
    assert source["captured_at"]
    assert len(source["records"]) > 0
    assert "tiempo de espera" in source["error"]
