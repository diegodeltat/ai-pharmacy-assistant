"""Evaluación end-to-end contra una API local o desplegada."""

import json
import os
from pathlib import Path
from typing import Any

import httpx


BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "evaluation_cases.json"
RESULTS_PATH = BASE_DIR / "evaluation_results.json"


def _check_sources(
    body: dict[str, Any],
    expected_source_type: str | None,
    require_sources: bool | None,
) -> tuple[bool, list[str]]:
    """Valida presencia y tipo de fuentes cuando el caso lo exige."""

    errors = []
    sources = body.get("sources", [])

    if require_sources is True and not sources:
        errors.append("Se esperaban fuentes, pero la respuesta no contiene ninguna.")

    if require_sources is False and sources:
        errors.append("No se esperaban fuentes, pero la respuesta contiene fuentes.")

    if expected_source_type:
        source_types = {
            source.get("source_type")
            for source in sources
            if isinstance(source, dict)
        }

        if expected_source_type not in source_types:
            errors.append(
                f"No se encontró source_type={expected_source_type}. "
                f"Recibidos: {sorted(source_types)}"
            )

    return not errors, errors


def _check_response_content(
    body: dict[str, Any],
    contains_any: list[str] | None,
    not_contains_any: list[str] | None,
) -> tuple[bool, list[str]]:
    """Hace validaciones simples sobre el texto de respuesta."""

    errors = []

    response_text = str(body.get("respuesta", "")).casefold()

    if not response_text.strip():
        errors.append("La respuesta está vacía.")

    if contains_any:
        expected_terms = [term.casefold() for term in contains_any]

        if not any(term in response_text for term in expected_terms):
            errors.append(
                "La respuesta no contiene ninguno de los términos esperados: "
                f"{contains_any}"
            )

    if not_contains_any:
        forbidden_terms = [term.casefold() for term in not_contains_any]

        found = [
            term
            for term in forbidden_terms
            if term in response_text
        ]

        if found:
            errors.append(
                f"La respuesta contiene términos no permitidos: {found}"
            )

    return not errors, errors


def _evaluate_case(
    case: dict[str, Any],
    body: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Evalúa una respuesta utilizando únicamente checks definidos por el caso."""

    errors = []

    expected_intent = case.get("expected_intent")
    if expected_intent is not None:
        actual_intent = body.get("intent")

        if actual_intent != expected_intent:
            errors.append(
                f"Intent esperado={expected_intent}, recibido={actual_intent}."
            )

    expected_blocked = case.get("expected_blocked")
    if expected_blocked is not None:
        actual_blocked = body.get("safety_blocked")

        if actual_blocked != expected_blocked:
            errors.append(
                f"safety_blocked esperado={expected_blocked}, "
                f"recibido={actual_blocked}."
            )

    sources_ok, source_errors = _check_sources(
        body=body,
        expected_source_type=case.get("expected_source_type"),
        require_sources=case.get("require_sources"),
    )

    content_ok, content_errors = _check_response_content(
        body=body,
        contains_any=case.get("contains_any"),
        not_contains_any=case.get("not_contains_any"),
    )

    errors.extend(source_errors)
    errors.extend(content_errors)

    passed = (
        not errors
        and sources_ok
        and content_ok
    )

    return passed, errors


def main() -> None:
    api_url = os.getenv(
        "API_BASE_URL",
        "http://127.0.0.1:8000",
    ).rstrip("/")

    cases = json.loads(
        CASES_PATH.read_text(encoding="utf-8")
    )

    results = []

    # conversation_id -> user_id real utilizado por la API.
    #
    # Casos que compartan conversation_id usan el mismo user_id,
    # permitiendo evaluar memoria multi-turno.
    conversations: dict[str, str] = {}

    with httpx.Client(timeout=40) as client:
        for index, case in enumerate(cases, start=1):
            case_id = case.get("id", f"CASE-{index:03d}")

            conversation_id = case.get(
                "conversation_id",
                case_id,
            )

            if conversation_id not in conversations:
                conversations[conversation_id] = (
                    f"eval-{conversation_id}"
                )

            user_id = conversations[conversation_id]

            print(
                f"[{index:02d}/{len(cases)}] "
                f"{case_id} | {case.get('category', 'sin-categoria')}"
            )

            try:
                response = client.post(
                    f"{api_url}/chat",
                    json={
                        "user_id": user_id,
                        "pregunta": case["question"],
                    },
                )

                response.raise_for_status()
                body = response.json()

                passed, errors = _evaluate_case(
                    case=case,
                    body=body,
                )

                results.append(
                    {
                        **case,
                        "user_id": user_id,
                        "passed": passed,
                        "errors": errors,
                        "response": body,
                    }
                )

                status = "PASS" if passed else "FAIL"
                print(f"  -> {status}")

                if errors:
                    for error in errors:
                        print(f"     - {error}")

            except Exception as error:
                results.append(
                    {
                        **case,
                        "user_id": user_id,
                        "passed": False,
                        "errors": [str(error)],
                    }
                )

                print(f"  -> ERROR: {error}")

    RESULTS_PATH.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total = len(results)
    passed = sum(
        1
        for result in results
        if result["passed"]
    )

    failed = total - passed

    by_category: dict[str, dict[str, int]] = {}

    for result in results:
        category = result.get(
            "category",
            "sin-categoria",
        )

        if category not in by_category:
            by_category[category] = {
                "total": 0,
                "passed": 0,
            }

        by_category[category]["total"] += 1

        if result["passed"]:
            by_category[category]["passed"] += 1

    print("\n" + "=" * 60)
    print("RESUMEN DE EVALUACIÓN")
    print("=" * 60)

    print(f"Total:     {total}")
    print(f"Aprobados: {passed}")
    print(f"Fallidos:  {failed}")

    if total:
        percentage = passed / total * 100
        print(f"Resultado: {percentage:.1f}%")

    print("\nPor categoría:")

    for category, summary in sorted(by_category.items()):
        print(
            f"- {category}: "
            f"{summary['passed']}/{summary['total']}"
        )

    print(f"\nDetalle completo: {RESULTS_PATH}")


if __name__ == "__main__":
    main()