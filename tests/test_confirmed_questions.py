import importlib
import json

import pytest
from fastapi.testclient import TestClient

from backend.question_discovery import (
    discover_questions,
    list_confirmed_questions,
    review_question_suggestion,
    update_confirmed_question_status,
)


@pytest.fixture
def suggestion(tmp_path):
    folder = tmp_path / "outputs" / "m1" / "r1"
    (folder / "transcript").mkdir(parents=True)
    (folder / "pipeline_status.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (folder / "transcript" / "aligned_transcript.json").write_text(
        json.dumps({"utterances": [{
            "id": 7, "text": "Когда запускаем пилот? Пока неизвестно.",
            "start": 1.0, "end": 5.0,
        }]}), encoding="utf-8"
    )
    result = discover_questions(
        tmp_path, "m1", "r1",
        lambda context: {"questions": [{
            "text": "Когда запускаем пилот?", "source_ids": [1]
        }]},
    )
    return tmp_path, result["items"][0]


def test_acceptance_is_idempotent_and_preserves_status(suggestion):
    root, item = suggestion
    identifier = item["suggestion_id"]
    first = review_question_suggestion(root, identifier, "accepted")
    question_id = first["question"]["question_id"]

    update_confirmed_question_status(root, question_id, "resolved")
    second = review_question_suggestion(root, identifier, "accepted")

    assert second["question"]["status"] == "resolved"
    assert len(list_confirmed_questions(root)) == 1
    assert list_confirmed_questions(root, status="open") == []
    assert list_confirmed_questions(root, kind="task") == []


def test_rejection_does_not_create_question(suggestion):
    root, item = suggestion
    review_question_suggestion(root, item["suggestion_id"], "rejected")
    assert list_confirmed_questions(root) == []
    with pytest.raises(ValueError):
        review_question_suggestion(root, item["suggestion_id"], "accepted")


def test_confirmed_question_is_available_to_both_review_modes(
    suggestion, monkeypatch
):
    from backend.memory_store import MemoryStore

    root, item = suggestion
    main = importlib.import_module("backend.main")
    monkeypatch.setattr(main, "ROOT", root)
    monkeypatch.setattr(MemoryStore, "sync", lambda self: {})

    def response(text):
        return {
            "question": text,
            "status": "answered",
            "answer": "20 ноября [1]",
            "sources": [{
                "id": 1, "text": "Запуск 20 ноября.",
                "meeting_id": "m2", "run_id": "r2",
                "utterance_id": "10",
            }],
            "retrieved_sources": [],
            "agent_trace": [],
            "stop_reason": "model_finished",
            "tool_calls": 1,
        }

    monkeypatch.setattr(
        main, "meeting_answer", lambda request: response(request.question)
    )
    monkeypatch.setattr(
        main, "run_live_agent", lambda root, text, **kwargs: response(text)
    )

    # The generator is stubbed; only the API's index-existence guard is needed.
    (root / "data" / "rag_index.sqlite3").touch()

    with TestClient(main.app) as client:
        accepted = client.post(
            f"/api/question-discoveries/{item['suggestion_id']}/accept"
        )
        assert accepted.status_code == 200
        question_id = accepted.json()["question"]["question_id"]

        questions = client.get(
            "/api/memory/items",
            params={"kind": "open_question", "status": "open"},
        ).json()["items"]
        assert any(q["question_id"] == question_id for q in questions)

        quick = client.post("/api/question-reviews/run", json={
            "question_ids": [question_id], "limit": 1,
        })
        assert quick.status_code == 200
        assert quick.json()["results"][0]["review"]["status"] == "candidate_answer"

        agent = client.post(
            "/api/question-agent/run", json={"question_id": question_id}
        )
        assert agent.status_code == 200
        assert agent.json()["agent"]["status"] == "answered"

        changed = client.patch(
            f"/api/memory/items/open_question/{question_id}",
            json={"status": "resolved"},
        )
        assert changed.status_code == 200
        assert changed.json()["status"] == "resolved"

        blocked = client.post(
            "/api/question-agent/run", json={"question_id": question_id}
        )
        assert blocked.status_code == 422


def test_missing_confirmed_question_does_not_create_database(tmp_path):
    assert list_confirmed_questions(tmp_path) == []
    assert update_confirmed_question_status(
        tmp_path, "discovered_missing", "resolved"
    ) is None
    assert not (tmp_path / "data").exists()
