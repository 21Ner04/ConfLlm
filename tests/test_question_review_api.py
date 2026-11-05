import importlib

import pytest
from fastapi.testclient import TestClient

from backend.memory_store import MemoryStore


@pytest.fixture
def api_environment(tmp_path, monkeypatch):
    main = importlib.import_module("backend.main")
    monkeypatch.setattr(main, "ROOT", tmp_path)

    # Keep the temporary seeded memory instead of scanning outputs.
    monkeypatch.setattr(MemoryStore, "sync", lambda self: {})

    with MemoryStore(tmp_path) as memory:
        with memory.connection:
            memory.connection.execute(
                """INSERT INTO open_questions
                (question_id, decision_id, text, owner, deadline,
                 status, meeting_id, run_id, source_item_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "test-q1", None, "When is the pilot?",
                    None, None, "open", "test-m1", "test-r1", "1",
                ),
            )

    calls = []

    def generate(request):
        calls.append(request)
        return {
            "status": "answered",
            "answer": "Friday was proposed. [1]",
            "sources": [{
                "id": 1,
                "meeting_id": "test-m2",
                "run_id": "test-r2",
                "utterance_id": 10,
                "start": 5.0,
                "text": "We propose Friday.",
            }],
            "metrics": {"total_tokens": 42},
        }

    monkeypatch.setattr(main, "meeting_answer", generate)

    with TestClient(main.app) as client:
        yield client, main, calls, tmp_path


def test_api_saves_candidate_and_keeps_question_open(api_environment):
    client, _, calls, root = api_environment

    response = client.post(
        "/api/question-reviews/run",
        json={"limit": 1, "question_ids": ["test-q1"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["created"] == 1
    assert data["errors"] == []
    assert data["results"][0]["review"]["status"] == "candidate_answer"
    assert len(calls) == 1
    assert calls[0].question == "When is the pilot?"
    assert calls[0].meeting_id is None

    history = client.get(
        "/api/question-reviews",
        params={"question_id": "test-q1"},
    )
    assert history.status_code == 200
    assert history.json()["count"] == 1
    assert history.json()["items"][0]["sources"][0]["utterance_id"] == 10

    with MemoryStore(root) as memory:
        question = memory.list_items(kind="open_question")[0]
        assert question["status"] == "open"


def test_api_deduplicates_repeated_results(api_environment):
    client, _, _, _ = api_environment
    payload = {"limit": 1, "question_ids": ["test-q1"]}

    first = client.post("/api/question-reviews/run", json=payload)
    second = client.post("/api/question-reviews/run", json=payload)

    assert first.status_code == second.status_code == 200
    assert first.json()["created"] == 1
    assert second.json()["created"] == 0
    assert client.get("/api/question-reviews").json()["count"] == 1


def test_api_rejects_unknown_question_without_model_call(api_environment):
    client, _, calls, _ = api_environment

    response = client.post(
        "/api/question-reviews/run",
        json={"question_ids": ["missing"]},
    )
    assert response.status_code == 422
    assert calls == []


def test_api_reports_generator_error_and_releases_lock(
    api_environment, monkeypatch
):
    from fastapi import HTTPException

    client, main, _, _ = api_environment

    def unavailable(request):
        raise HTTPException(503, "private service details")

    monkeypatch.setattr(main, "meeting_answer", unavailable)

    for _ in range(2):
        response = client.post(
            "/api/question-reviews/run",
            json={"question_ids": ["test-q1"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "partial"
        assert data["created"] == 0
        assert data["errors"][0]["http_status"] == 503
        assert "private service details" not in response.text

    assert client.get("/api/question-reviews").json()["count"] == 0


def test_api_empty_memory_does_not_call_model(api_environment):
    client, _, calls, root = api_environment

    with MemoryStore(root) as memory:
        with memory.connection:
            memory.connection.execute("DELETE FROM open_questions")

    response = client.post("/api/question-reviews/run", json={})
    assert response.status_code == 200
    assert response.json()["attempted"] == 0
    assert response.json()["results"] == []
    assert calls == []


@pytest.mark.parametrize("limit", [0, 4, True])
def test_api_rejects_invalid_run_limit(api_environment, limit):
    client, _, calls, _ = api_environment
    response = client.post(
        "/api/question-reviews/run",
        json={"limit": limit},
    )
    assert response.status_code == 422
    assert calls == []
