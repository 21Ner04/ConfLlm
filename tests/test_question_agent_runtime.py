from types import SimpleNamespace

import pytest

from backend.question_agent_runtime import make_planner, make_tools
from backend.memory_store import MemoryStore


def test_planner_calls_existing_chat_api_and_parses_action():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content='{"action":"search_memory","query":"pilot"}'
                ),
            )],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )
    calls = []
    action = make_planner(client, "test-model", calls)({
        "question": "When?",
        "evidence": [],
    })
    assert action["action"] == "search_memory"
    assert captured["model"] == "test-model"
    assert calls[0]["total_tokens"] == 15


def test_planner_rejects_truncated_response():
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content="{}"),
                )],
                usage=None,
            )
        )
    ))
    with pytest.raises(ValueError):
        make_planner(client, "test-model", [])({})


def test_vector_tools_apply_source_filters(tmp_path, monkeypatch):
    import backend.question_agent_runtime as runtime

    calls = []

    class Index:
        def __init__(self, root):
            assert root == tmp_path

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def query(self, query, **kwargs):
            calls.append((query, kwargs))
            return []

    monkeypatch.setattr(runtime, "LocalVectorIndex", Index)
    tools = make_tools(tmp_path)
    tools["search_transcripts"]("pilot")
    tools["search_documents"]("approval")
    assert calls[0][1]["source_types"] == ["utterance"]
    assert calls[1][1]["source_types"] == ["document"]


def test_memory_tool_preserves_reference_and_handles_cyrillic(tmp_path):
    with MemoryStore(tmp_path) as memory:
        with memory.connection:
            memory.connection.execute(
                """INSERT INTO evidence_refs
                (evidence_id, entity_type, entity_id, meeting_id, run_id,
                 source_item_id, utterance_ids_json, source_url, text, kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "e1", "decision", "d1", "m1", "r1", "1",
                    "[10]", "meeting://m1/r1/utterance/10",
                    "Запуск пилота согласован на 20 ноября.", "decision",
                ),
            )

    result = make_tools(tmp_path)["search_memory"]("ПИЛОТА")
    assert len(result) == 1
    assert result[0]["source_type"] == "memory_decision"
    assert result[0]["source_url"].endswith("/utterance/10")
    assert make_tools(tmp_path)["search_memory"]("несуществующийтермин") == []


def test_agent_run_is_saved_in_separate_database(tmp_path):
    import json
    import sqlite3
    from contextlib import closing
    from backend.question_agent_runtime import save_agent_run

    saved = save_agent_run(tmp_path, {
        "status": "incomplete",
        "stop_reason": "tool_limit",
        "agent_trace": [{"action": "search_memory", "status": "ok"}],
    })
    with closing(sqlite3.connect(
        tmp_path / "data" / "question_agent_runs.sqlite3"
    )) as connection:
        raw = connection.execute(
            "SELECT payload_json FROM agent_runs WHERE run_id=?",
            (saved["agent_run_id"],),
        ).fetchone()[0]
    assert json.loads(raw)["stop_reason"] == "tool_limit"
