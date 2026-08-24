from evaluation.evaluate import _evaluate_case, _parse_bool


def test_rag_case_rejects_wrong_source_and_missing_citation():
    case = {
        "expected_intent": "medication",
        "require_sources": True,
        "expected_source_type": "rag",
        "expected_top1_title": "Amoxicillin",
        "expected_source_titles": ["Amoxicillin"],
        "require_citations": True,
        "expected_abstention": False,
    }
    body = {
        "intent": "medication",
        "respuesta": "Información general sin referencia.",
        "sources": [
            {"source_type": "rag", "title": "Loratadine"},
        ],
    }

    passed, errors = _evaluate_case(case, body)

    assert passed is False
    assert any("Top 1 esperado" in error for error in errors)
    assert any("cita" in error for error in errors)


def test_no_answer_case_requires_abstention_without_sources():
    case = {
        "expected_intent": "medication",
        "require_sources": False,
        "require_citations": False,
        "expected_abstention": True,
        "expected_clarification": True,
    }
    body = {
        "intent": "medication",
        "respuesta": (
            "No encontré una ficha coincidente. Indica el nombre exacto "
            "del medicamento."
        ),
        "sources": [],
    }

    passed, errors = _evaluate_case(case, body)

    assert passed is True
    assert errors == []


def test_evaluation_rerank_boolean_parser():
    assert _parse_bool("true") is True
    assert _parse_bool("sí") is True
    assert _parse_bool("false") is False
