import importlib

import pytest
from fastapi.testclient import TestClient

from backend.question_agent_runtime import list_agent_runs, save_agent_run


def test_missing_history_does_not_create_database(tmp_path):
    assert list_agent_runs(tmp_path) == []
    assert not (tmp_path / "data").exists()


def test_history_filters_and_returns_newest_insert_first(tmp_path):
    first = save_agent_run(tmp_path, {
        "question_id": "q1",
        "question": "First",
        "status": "incomplete",
        "stop_reason": "tool_limit",
        "agent_trace": [],
    })
    second = save_agent_run(tmp_path, {
        "question_id": "q2",
        "question": "Second",
        "status": "error",
        "stop_reason": "planner_failed",
        "agent_trace": [],
    })

    assert list_agent_runs(tmp_path, limit=1)[0]["agent_run_id"] == second["agent_run_id"]
    filtered = list_agent_runs(tmp_path, question_id="q1")
    assert len(filtered) == 1
    assert filtered[0]["agent_run_id"] == first["agent_run_id"]
    assert list_agent_runs(tmp_path, question_id="missing") == []


@pytest.mark.parametrize("limit", [0, 101, True])
def test_invalid_storage_limit_is_rejected(tmp_path, limit):
    with pytest.raises(ValueError):
        list_agent_runs(tmp_path, limit=limit)


def test_history_api_reads_saved_run(tmp_path, monkeypatch):
    main = importlib.import_module("backend.main")
    monkeypatch.setattr(main, "ROOT", tmp_path)
    saved = save_agent_run(tmp_path, {
        "question_id": "q1",
        "question": "When?",
        "status": "insufficient_sources",
        "stop_reason": "model_finished",
        "agent_trace": [{"step": 1, "action": "finish", "status": "ok"}],
    })

    with TestClient(main.app) as client:
        response = client.get(
            "/api/question-agent/runs",
            params={"question_id": "q1", "limit": 20},
        )
        assert response.status_code == 200
        assert response.json()["count"] == 1
        assert response.json()["items"][0]["agent_run_id"] == saved["agent_run_id"]

        assert client.get(
            "/api/question-agent/runs", params={"limit": 101}
        ).status_code == 422
