"""Evaluación end-to-end contra una API local o desplegada."""

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "evaluation_cases.json"
RESULTS_PATH = BASE_DIR / "evaluation_results.json"


def _parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on", "si", "sí"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("Usa true o false.")


def _check_sources(
    body: dict[str, Any],
    expected_source_type: str | None,
    require_sources: bool | None,
    expected_source_titles: list[str] | None,
    expected_top1_title: str | None,
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

    actual_titles = [
        str(source.get("title", "")).casefold()
        for source in sources
        if isinstance(source, dict)
    ]
    if expected_source_titles:
        missing_titles = [
            title
            for title in expected_source_titles
            if title.casefold() not in actual_titles
        ]
        if missing_titles:
            errors.append(
                "No se recuperaron las fuentes esperadas: "
                f"{missing_titles}. Recibidas: {actual_titles}"
            )

    if expected_top1_title:
        actual_top1 = actual_titles[0] if actual_titles else None
        if actual_top1 != expected_top1_title.casefold():
            errors.append(
                f"Top 1 esperado={expected_top1_title}, recibido={actual_top1}."
            )

    return not errors, errors


def _check_response_content(
    body: dict[str, Any],
    contains_any: list[str] | None,
    not_contains_any: list[str] | None,
    require_citations: bool | None,
    expected_abstention: bool | None,
    expected_clarification: bool | None,
) -> tuple[bool, list[str]]:
    """Hace validaciones simples sobre el texto de respuesta."""

    errors = []

    raw_response = str(body.get("respuesta", ""))
    response_text = raw_response.casefold()

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

    citations = [
        int(value)
        for value in re.findall(
            r"\[Fuente\s+(\d+)\]",
            raw_response,
            flags=re.IGNORECASE,
        )
    ]
    if require_citations is True and not citations:
        errors.append("Se esperaba al menos una cita [Fuente N] válida.")
    if require_citations is False and citations:
        errors.append("No se esperaban citas en la respuesta.")
    source_count = len(body.get("sources", []))
    invalid_citations = [
        index for index in citations if index < 1 or index > source_count
    ]
    if invalid_citations:
        errors.append(
            f"Las citas no corresponden a fuentes devueltas: {invalid_citations}."
        )

    abstention_markers = (
        "no se encontró información",
        "no se encontro informacion",
        "no encontré una ficha",
        "no encontre una ficha",
        "no fue posible generar una respuesta",
    )
    abstained = any(marker in response_text for marker in abstention_markers)
    if expected_abstention is True and not abstained:
        errors.append("Se esperaba una abstención explícita.")
    if expected_abstention is False and abstained:
        errors.append("No se esperaba una abstención para este caso.")

    clarification_markers = (
        "indica el nombre exacto",
        "necesito el nombre",
        "especifica el nombre",
    )
    clarified = any(marker in response_text for marker in clarification_markers)
    if expected_clarification is True and not clarified:
        errors.append("Se esperaba solicitar el nombre del medicamento.")

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
        expected_source_titles=case.get("expected_source_titles"),
        expected_top1_title=case.get("expected_top1_title"),
    )

    content_ok, content_errors = _check_response_content(
        body=body,
        contains_any=case.get("contains_any"),
        not_contains_any=case.get("not_contains_any"),
        require_citations=case.get("require_citations"),
        expected_abstention=case.get("expected_abstention"),
        expected_clarification=case.get("expected_clarification"),
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
    parser = argparse.ArgumentParser(description="Ejecuta la evaluación end-to-end")
    parser.add_argument(
        "--rerank",
        nargs="?",
        const=True,
        type=_parse_bool,
        default=None,
        help=(
            "Activa reranking. Acepta true/false; sin valor equivale a true. "
            "Si se omite, el backend usa su configuración por defecto."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api_url = os.getenv(
        "API_BASE_URL",
        "http://127.0.0.1:8000",
    ).rstrip("/")
    if args.output is not None:
        results_path = args.output
    elif args.rerank is True:
        results_path = BASE_DIR / "evaluation_results_rerank_on.json"
    elif args.rerank is False:
        results_path = BASE_DIR / "evaluation_results_rerank_off.json"
    else:
        results_path = RESULTS_PATH

    cases = json.loads(
        CASES_PATH.read_text(encoding="utf-8")
    )

    results = []

    # conversation_id -> user_id real utilizado por la API.
    #
    # Casos que compartan conversation_id usan el mismo user_id,
    # permitiendo evaluar memoria multi-turno.
    conversations: dict[str, str] = {}
    mode_label = (
        "rerank-on"
        if args.rerank is True
        else "rerank-off"
        if args.rerank is False
        else "backend-default"
    )

    with httpx.Client(timeout=40) as client:
        for index, case in enumerate(cases, start=1):
            case_id = case.get("id", f"CASE-{index:03d}")

            conversation_id = case.get(
                "conversation_id",
                case_id,
            )

            if conversation_id not in conversations:
                conversations[conversation_id] = (
                    f"eval-{mode_label}-{conversation_id}"
                )

            user_id = conversations[conversation_id]

            print(
                f"[{index:02d}/{len(cases)}] "
                f"{case_id} | {case.get('category', 'sin-categoria')}"
            )

            try:
                started_at = time.perf_counter()
                request_body = {
                    "user_id": user_id,
                    "pregunta": case["question"],
                }
                if args.rerank is not None:
                    request_body["rerank"] = args.rerank

                response = client.post(
                    f"{api_url}/chat",
                    json=request_body,
                )

                response.raise_for_status()
                body = response.json()
                latency_ms = (time.perf_counter() - started_at) * 1000

                passed, errors = _evaluate_case(
                    case=case,
                    body=body,
                )
                if (
                    args.rerank is not None
                    and body.get("rerank_enabled") is not args.rerank
                ):
                    passed = False
                    errors.append(
                        "El backend no aplicó el modo rerank solicitado: "
                        f"esperado={args.rerank}, "
                        f"recibido={body.get('rerank_enabled')}."
                    )

                results.append(
                    {
                        **case,
                        "user_id": user_id,
                        "evaluation_rerank": args.rerank,
                        "latency_ms": round(latency_ms, 2),
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
                latency_ms = (time.perf_counter() - started_at) * 1000
                results.append(
                    {
                        **case,
                        "user_id": user_id,
                        "evaluation_rerank": args.rerank,
                        "latency_ms": round(latency_ms, 2),
                        "passed": False,
                        "errors": [str(error)],
                    }
                )

                print(f"  -> ERROR: {error}")

    results_path.write_text(
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
    print(f"Reranking solicitado: {args.rerank}")

    print(f"Total:     {total}")
    print(f"Aprobados: {passed}")
    print(f"Fallidos:  {failed}")

    if total:
        percentage = passed / total * 100
        print(f"Resultado: {percentage:.1f}%")

    latencies = [
        float(result["latency_ms"])
        for result in results
        if result.get("latency_ms") is not None
    ]
    if latencies:
        ordered = sorted(latencies)
        p95_index = max(0, int(len(ordered) * 0.95 + 0.9999) - 1)
        print(f"Latencia p50: {statistics.median(ordered):.1f} ms")
        print(f"Latencia p95: {ordered[p95_index]:.1f} ms")

    print("\nPor categoría:")

    for category, summary in sorted(by_category.items()):
        print(
            f"- {category}: "
            f"{summary['passed']}/{summary['total']}"
        )

    print(f"\nDetalle completo: {results_path}")


if __name__ == "__main__":
    main()
