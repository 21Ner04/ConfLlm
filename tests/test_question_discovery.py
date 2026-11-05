import json

import pytest

from backend.question_discovery import (
    discover_questions,
    list_question_suggestions,
    load_discovery_context,
)


def create_run(root, utterances=None, status="completed"):
    folder = root / "outputs" / "m1" / "r1"
    (folder / "transcript").mkdir(parents=True)
    (folder / "pipeline_status.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    (folder / "transcript" / "aligned_transcript.json").write_text(
        json.dumps({
            "utterances": utterances if utterances is not None else [{
                "id": 7,
                "text": "Какой бюджет согласован? Пока не согласован.",
                "start": 1,
                "end": 5,
            }]
        }),
        encoding="utf-8",
    )


def extract(context):
    return {"questions": [{
        "text": "Какой бюджет согласован?",
        "source_ids": [1],
    }]}


def test_discovery_saves_proposal_with_original_source(tmp_path):
    create_run(tmp_path)
    result = discover_questions(tmp_path, "m1", "r1", extract)
    assert result["created"] == 1
    item = result["items"][0]
    assert item["status"] == "proposed"
    assert item["requires_human_review"] is True
    assert item["sources"][0]["utterance_id"] == 7
    assert item["coverage"]["partial"] is False
    assert not (tmp_path / "data" / "meeting_memory.sqlite3").exists()


def test_identical_discovery_is_deduplicated(tmp_path):
    create_run(tmp_path)
    discover_questions(tmp_path, "m1", "r1", extract)
    result = discover_questions(tmp_path, "m1", "r1", extract)
    assert result["created"] == 0
    assert len(list_question_suggestions(tmp_path)) == 1
    assert list_question_suggestions(tmp_path, meeting_id="missing") == []


def test_unknown_source_is_rejected_before_saving(tmp_path):
    create_run(tmp_path)
    with pytest.raises(ValueError):
        discover_questions(tmp_path, "m1", "r1", lambda context: {
            "questions": [{"text": "Question?", "source_ids": [99]}]
        })
    assert not (tmp_path / "data" / "question_discoveries.sqlite3").exists()


def test_long_transcript_is_explicitly_partial(tmp_path):
    create_run(tmp_path, [
        {"id": value, "text": "Question about the pilot."}
        for value in range(50)
    ])
    context = load_discovery_context(tmp_path, "m1", "r1")
    assert len(context["sources"]) == 40
    assert context["coverage"]["partial"] is True


def test_long_utterance_marks_partial_coverage(tmp_path):
    create_run(tmp_path, [{"id": 1, "text": "x" * 2000}])
    context = load_discovery_context(tmp_path, "m1", "r1")
    assert context["sources"][0]["truncated"] is True
    assert context["coverage"]["partial"] is True


def test_empty_transcript_does_not_call_model(tmp_path):
    create_run(tmp_path, [])

    def forbidden(context):
        raise AssertionError("Model must not be called")

    result = discover_questions(tmp_path, "m1", "r1", forbidden)
    assert result["status"] == "no_text"


def test_incomplete_run_is_rejected(tmp_path):
    create_run(tmp_path, status="running")
    with pytest.raises(ValueError):
        load_discovery_context(tmp_path, "m1", "r1")


def test_path_traversal_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_discovery_context(tmp_path, "../outside", "r1")


def test_empty_model_result_is_allowed(tmp_path):
    create_run(tmp_path)
    result = discover_questions(
        tmp_path, "m1", "r1", lambda context: {"questions": []}
    )
    assert result["items"] == []
    assert result["created"] == 0


def test_reading_missing_history_does_not_create_data(tmp_path):
    assert list_question_suggestions(tmp_path) == []
    assert not (tmp_path / "data").exists()
