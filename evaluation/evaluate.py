"""Evaluación end-to-end contra una API local o desplegada."""

import json
import os
import uuid
from pathlib import Path

import httpx


BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "evaluation_cases.json"
RESULTS_PATH = BASE_DIR / "evaluation_results.json"


def main() -> None:
    api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    results = []
    with httpx.Client(timeout=40) as client:
        for case in cases:
            try:
                response = client.post(
                    f"{api_url}/chat",
                    json={
                        "user_id": f"eval-{uuid.uuid4()}",
                        "pregunta": case["question"],
                    },
                )
                response.raise_for_status()
                body = response.json()
                passed = (
                    body["intent"] == case["expected_intent"]
                    and body["safety_blocked"] == case["expected_blocked"]
                )
                results.append({**case, "passed": passed, "response": body})
            except Exception as error:
                results.append({**case, "passed": False, "error": str(error)})

    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    passed = sum(result["passed"] for result in results)
    print(f"Evaluación: {passed}/{len(results)} casos aprobados")
    print(f"Detalle: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
