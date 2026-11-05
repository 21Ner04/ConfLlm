import json

from backend.question_agent_runtime import expand_transcript_results


def write_run(root, meeting, status="completed"):
    folder = root / "outputs" / meeting / "r1"
    (folder / "transcript").mkdir(parents=True)
    (folder / "pipeline_status.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    (folder / "transcript" / "aligned_transcript.json").write_text(
        json.dumps({"utterances": [
            {
                "id": 39, "speaker": "speaker_1",
                "text": "Кем ты видишь себя через 5 лет?",
                "start": 588.0, "end": 590.0,
            },
            {
                "id": 40, "speaker": "speaker_2",
                "text": "Хочу заниматься тем же, только для души.",
                "start": 590.0, "end": 598.0,
            },
            {
                "id": 41, "speaker": "speaker_1",
                "text": "О каком агенте ты мечтаешь?",
                "start": 598.0, "end": 600.0,
            },
        ]}, ensure_ascii=False),
        encoding="utf-8",
    )


def hit(meeting):
    return {
        "meeting_id": meeting,
        "run_id": "r1",
        "utterance_id": "39",
        "source_type": "utterance",
        "text": "Кем ты видишь себя через 5 лет?",
    }


def test_original_question_window_includes_following_answer(tmp_path):
    write_run(tmp_path, "original")
    result = expand_transcript_results(
        tmp_path, [], anchors=[hit("original")]
    )
    assert [item["utterance_id"] for item in result] == [39, 40]
    assert "для души" in result[1]["text"]
    assert result[1]["source_url"].endswith("/utterance/40")


def test_duplicate_copies_do_not_fill_context_with_questions(tmp_path):
    for name in ("original", "copy1", "copy2"):
        write_run(tmp_path, name)

    result = expand_transcript_results(
        tmp_path,
        [hit("copy1"), hit("copy2")],
        anchors=[hit("original")],
    )
    assert len(result) == 2
    assert all(item["meeting_id"] == "original" for item in result)
    assert any(item["utterance_id"] == 40 for item in result)


def test_failed_anchor_is_not_loaded(tmp_path):
    write_run(tmp_path, "failed_run", status="failed")
    assert expand_transcript_results(
        tmp_path, [], anchors=[hit("failed_run")]
    ) == []


def test_missing_transcript_keeps_existing_indexed_hit(tmp_path):
    result = expand_transcript_results(tmp_path, [hit("missing")])
    assert len(result) == 1
    assert result[0]["text"] == hit("missing")["text"]


def test_path_traversal_anchor_is_ignored(tmp_path):
    assert expand_transcript_results(
        tmp_path, [], anchors=[hit("../outside")]
    ) == []
