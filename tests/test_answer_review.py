import pytest

from backend.answer_review import apply_claim_checks


def draft():
    return {
        "status": "answered",
        "stop_reason": "model_finished",
        "question": "Кем ты видишь себя через пять лет?",
        "answer": "Продолжать дело. [2]\n\nМечтает об агенте. [6]",
        "claims": [
            {"text": "Продолжать дело.", "source_ids": [2]},
            {"text": "Мечтает об агенте.", "source_ids": [6]},
        ],
        "sources": [
            {"id": 2, "text": "Хочу заниматься тем же, для души."},
            {"id": 6, "text": "Мечтаю об агенте для обещаний."},
        ],
        "agent_trace": [],
    }


def test_irrelevant_claim_is_removed_without_rewriting_answer():
    original = draft()
    result = apply_claim_checks(original, {"checks": [
        {"claim_id": 1, "relevant": True, "supported": True},
        {"claim_id": 2, "relevant": False, "supported": True},
    ]})
    assert result["status"] == "answered"
    assert result["answer"] == "Продолжать дело. [2]"
    assert [source["id"] for source in result["sources"]] == [2]
    assert result["agent_trace"][-1]["removed"] == 1
    assert len(original["claims"]) == 2


def test_all_rejected_is_incomplete_not_no_answer():
    result = apply_claim_checks(draft(), {"checks": [
        {"claim_id": 1, "relevant": True, "supported": False},
        {"claim_id": 2, "relevant": False, "supported": True},
    ]})
    assert result["status"] == "incomplete"
    assert result["stop_reason"] == "answer_review_rejected"
    assert result["answer"] == ""
    assert result["sources"] == []


def test_missing_claim_check_is_rejected():
    with pytest.raises(ValueError):
        apply_claim_checks(draft(), {"checks": [
            {"claim_id": 1, "relevant": True, "supported": True},
        ]})


def test_duplicate_claim_check_is_rejected():
    with pytest.raises(ValueError):
        apply_claim_checks(draft(), {"checks": [
            {"claim_id": 1, "relevant": True, "supported": True},
            {"claim_id": 1, "relevant": True, "supported": True},
        ]})
