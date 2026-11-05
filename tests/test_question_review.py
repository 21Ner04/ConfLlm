import pytest

from backend.question_review import QuestionReviewStore, review_questions


def question(identifier="q1", status="open", text="When is the pilot?"):
    return {
        "kind": "open_question",
        "question_id": identifier,
        "text": text,
        "status": status,
        "meeting_id": "m1",
        "run_id": "r1",
    }


def answer(text="A pilot date was proposed."):
    return {
        "status": "answered",
        "answer": text,
        "sources": [{
            "id": 1,
            "meeting_id": "m2",
            "run_id": "r2",
            "utterance_id": 10,
            "text": "We propose Friday.",
        }],
        "metrics": {"total_tokens": 42},
    }


def test_candidate_preserves_sources_and_does_not_resolve_question(tmp_path):
    original = question()
    result = review_questions(tmp_path, [original], lambda _: answer())
    candidate = result["results"][0]["review"]
    assert candidate["status"] == "candidate_answer"
    assert candidate["requires_human_review"] is True
    assert candidate["sources"][0]["utterance_id"] == 10
    assert original["status"] == "open"
    with QuestionReviewStore(tmp_path) as store:
        assert store.list_reviews()[0] == candidate


def test_repeated_result_is_deduplicated_despite_metrics(tmp_path):
    first = review_questions(tmp_path, [question()], lambda _: answer())
    changed_metrics = answer()
    changed_metrics["metrics"] = {"total_tokens": 99}
    second = review_questions(
        tmp_path, [question()], lambda _: changed_metrics
    )
    assert first["created"] == 1
    assert second["created"] == 0
    with QuestionReviewStore(tmp_path) as store:
        assert len(store.list_reviews()) == 1


def test_changed_answer_creates_another_proposal(tmp_path):
    review_questions(tmp_path, [question()], lambda _: answer())
    result = review_questions(
        tmp_path, [question()], lambda _: answer("A different proposal.")
    )
    assert result["created"] == 1
    with QuestionReviewStore(tmp_path) as store:
        assert len(store.list_reviews("q1")) == 2
        assert store.list_reviews("missing") == []


@pytest.mark.parametrize("status", ["no_sources", "insufficient_sources"])
def test_no_evidence_does_not_become_candidate(tmp_path, status):
    result = review_questions(
        tmp_path, [question()],
        lambda _: {"status": status, "answer": "Not enough evidence.", "sources": []},
    )
    assert result["results"][0]["review"]["status"] == status


def test_filters_status_kind_ids_and_limits_calls(tmp_path):
    calls = []
    questions = [
        question("q1"),
        question("q2", status="resolved"),
        question("q3", status="in_progress"),
        {**question("q4"), "kind": "task"},
    ]

    def generate(text):
        calls.append(text)
        return answer()

    result = review_questions(
        tmp_path, questions, generate, limit=1, question_ids=["q3"]
    )
    assert result["eligible"] == 1
    assert result["results"][0]["review"]["question_id"] == "q3"
    assert len(calls) == 1

    limited = review_questions(tmp_path, questions, generate, limit=1)
    assert limited["eligible"] == 2
    assert limited["attempted"] == 1
    assert limited["remaining"] == 1


def test_failure_is_reported_and_next_question_is_processed(tmp_path):
    def generate(text):
        if text == "Broken":
            raise RuntimeError("private exception details")
        return answer()

    result = review_questions(
        tmp_path,
        [question("q1", text="Broken"), question("q2")],
        generate,
        limit=2,
    )
    assert result["status"] == "partial"
    assert result["errors"] == [{"question_id": "q1", "code": "review_failed"}]
    assert result["results"][0]["review"]["question_id"] == "q2"
    assert "private exception details" not in str(result)


def test_answer_without_sources_is_not_saved(tmp_path):
    invalid = answer()
    invalid["sources"] = []
    result = review_questions(tmp_path, [question()], lambda _: invalid)
    assert result["created"] == 0
    assert result["errors"][0]["code"] == "review_failed"
    with QuestionReviewStore(tmp_path) as store:
        assert store.list_reviews() == []


def test_invalid_question_does_not_call_model(tmp_path):
    def forbidden(_):
        raise AssertionError("Model must not be called")

    result = review_questions(
        tmp_path, [question(text=" " * 10)], forbidden
    )
    assert result["errors"][0]["code"] == "invalid_question"


def test_invalid_limit_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        review_questions(tmp_path, [], lambda _: answer(), limit=4)
