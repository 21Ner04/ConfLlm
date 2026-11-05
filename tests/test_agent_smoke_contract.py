from evals.smoke_question_agent import (
    check_budget_result,
    check_missing_data_result,
)


def budget_answer(text):
    return {
        "status": "answered",
        "answer": text,
        "sources": [{
            "meeting_id": "demo_original",
            "utterance_id": 6,
            "text": "Бюджет не утверждён. Точной суммы в рублях нет.",
        }],
    }


def test_grounded_negative_budget_answer_is_acceptable():
    assert check_budget_result(budget_answer(
        "Бюджет не утверждён, точной суммы нет. [2]"
    )) == []


def test_invented_budget_amount_is_rejected():
    assert check_budget_result(budget_answer(
        "Бюджет не утверждён, предполагается 500000 рублей. [2]"
    ))


def test_budget_answer_without_matching_source_is_rejected():
    result = budget_answer("Бюджет не утверждён. [2]")
    result["sources"] = []
    assert check_budget_result(result)


def test_unrelated_budget_answer_is_rejected():
    assert check_budget_result(budget_answer(
        "Пилот запускается в ноябре. [2]"
    ))


def test_missing_data_still_requires_refusal():
    assert check_missing_data_result({
        "status": "insufficient_sources", "sources": []
    }) == []

    assert check_missing_data_result({
        "status": "answered",
        "answer": "Выручка неизвестна.",
        "sources": [],
    })
