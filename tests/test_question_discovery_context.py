import pytest

from backend.question_discovery import (
    DiscoveryContextError,
    extract_discovery_batches,
)


def context():
    return {
        "sources": [
            {"id": 1, "text": "Первый вопрос о бюджете."},
            {"id": 2, "text": "Второй вопрос о сроке."},
        ],
        "coverage": {"partial": False},
    }


def test_overflow_splits_batch_and_preserves_source_ids():
    data = context()
    calls = []

    def request(batch):
        calls.append([item["id"] for item in batch])
        if len(batch) > 1:
            raise DiscoveryContextError("too large")
        identifier = batch[0]["id"]
        return {"questions": [{
            "text": f"Вопрос {identifier}?",
            "source_ids": [identifier],
        }]}

    result = extract_discovery_batches(data, request)
    assert calls == [[1, 2], [1], [2]]
    assert [q["source_ids"] for q in result["questions"]] == [[1], [2]]
    assert data["coverage"]["partial"] is True
    assert data["coverage"]["cross_chunk_validation_pending"] is True


def test_fitting_batch_does_not_split():
    data = context()
    result = extract_discovery_batches(
        data, lambda batch: {"questions": []}
    )
    assert result == {"questions": []}
    assert data["coverage"]["llm_calls"] == 1
    assert data["coverage"]["chunked"] is False
    assert data["coverage"]["partial"] is False


def test_unrelated_error_is_not_retried():
    calls = []

    def request(batch):
        calls.append(batch)
        raise RuntimeError("service unavailable")

    with pytest.raises(RuntimeError, match="service unavailable"):
        extract_discovery_batches(context(), request)
    assert len(calls) == 1


def test_unknown_source_from_another_batch_is_rejected():
    def request(batch):
        if len(batch) > 1:
            raise DiscoveryContextError()
        return {"questions": [{
            "text": "Вопрос?",
            "source_ids": [99],
        }]}

    with pytest.raises(ValueError):
        extract_discovery_batches(context(), request)


def test_call_limit_stops_recursive_splitting():
    def request(batch):
        raise DiscoveryContextError()

    with pytest.raises(RuntimeError, match="call limit"):
        extract_discovery_batches(context(), request, max_calls=1)


def test_single_long_source_can_be_split():
    data = {
        "sources": [{"id": 7, "text": "x" * 1000}],
        "coverage": {"partial": False},
    }

    def request(batch):
        if len(batch[0]["text"]) > 500:
            raise DiscoveryContextError()
        return {"questions": [{
            "text": "Вопрос?",
            "source_ids": [7],
        }]}

    result = extract_discovery_batches(data, request)
    assert len(result["questions"]) == 1
    assert result["questions"][0]["source_ids"] == [7]
    assert data["coverage"]["partial"] is True
