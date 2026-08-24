"""Calibra retrieval RAG con resultados reproducibles por threshold y top-k."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.rag.entity_resolver import resolve_medication_entity
from app.rag.retriever import retrieve_medications


BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "evaluation_cases.json"
RESULTS_PATH = BASE_DIR / "rag_calibration_results.json"


def _parse_numbers(raw: str, cast: type) -> list:
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def _evaluate_configuration(
    cases: list[dict[str, Any]],
    threshold: float,
    top_k: int,
) -> dict[str, Any]:
    details = []
    positive_total = 0
    positive_hit1 = 0
    negative_total = 0
    negative_abstained = 0

    for case in cases:
        expected_title = case.get("expected_top1_title")
        entity = resolve_medication_entity(case["question"])
        results = retrieve_medications(
            case["question"],
            expected_drug_name=entity if expected_title else None,
            top_k=top_k,
            score_threshold=threshold,
        )
        titles = [result.title for result in results]
        scores = [result.score for result in results]

        if expected_title:
            positive_total += 1
            passed = bool(titles and titles[0].casefold() == expected_title.casefold())
            positive_hit1 += int(passed)
        else:
            negative_total += 1
            passed = not results
            negative_abstained += int(passed)

        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_top1_title": expected_title,
                "resolved_entity": entity,
                "titles": titles,
                "scores": scores,
                "passed": passed,
            }
        )

    return {
        "threshold": threshold,
        "top_k": top_k,
        "hit_at_1": (
            positive_hit1 / positive_total if positive_total else None
        ),
        "negative_abstention_rate": (
            negative_abstained / negative_total if negative_total else None
        ),
        "positive": {"passed": positive_hit1, "total": positive_total},
        "negative": {"passed": negative_abstained, "total": negative_total},
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra el retrieval de Qdrant")
    parser.add_argument(
        "--thresholds",
        default="0.30,0.40,0.45,0.50,0.60,0.70,0.75",
    )
    parser.add_argument("--top-k", default="1,2,4")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    thresholds = _parse_numbers(args.thresholds, float)
    top_k_values = _parse_numbers(args.top_k, int)
    all_cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    rag_cases = [
        case
        for case in all_cases
        if case.get("category") in {"rag", "rag_no_answer"}
    ]

    configurations = [
        _evaluate_configuration(rag_cases, threshold, top_k)
        for threshold in thresholds
        for top_k in top_k_values
    ]
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "case_count": len(rag_cases),
        "configurations": configurations,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ranked = sorted(
        configurations,
        key=lambda item: (
            item["negative_abstention_rate"] or 0,
            item["hit_at_1"] or 0,
            -item["top_k"],
        ),
        reverse=True,
    )
    print(f"Casos evaluados: {len(rag_cases)}")
    for item in ranked:
        print(
            f"threshold={item['threshold']:.2f} top_k={item['top_k']} "
            f"hit@1={item['hit_at_1']:.1%} "
            f"abstención negativa={item['negative_abstention_rate']:.1%}"
        )
    print(f"Detalle: {args.output}")


if __name__ == "__main__":
    main()
