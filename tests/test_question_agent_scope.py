import json

from backend.question_agent_runtime import make_tools


def test_source_dialogue_has_no_cross_meeting_search_tools(tmp_path):
    folder = tmp_path / "outputs" / "original" / "r1"
    (folder / "transcript").mkdir(parents=True)
    (folder / "pipeline_status.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (folder / "transcript" / "aligned_transcript.json").write_text(
        json.dumps({"utterances": [
            {"id": 39, "text": "Кем ты видишь себя через 5 лет?"},
            {"id": 40, "text": "Хочу продолжать своё дело для души."},
            {"id": 41, "text": "О каком агенте ты мечтаешь?"},
            {"id": 42, "text": "Об агенте для обещаний."},
        ]}, ensure_ascii=False),
        encoding="utf-8",
    )

    tools = make_tools(tmp_path, question_context={
        "answer_scope": "source_dialogue",
        "evidence": [{
            "meeting_id": "original",
            "run_id": "r1",
            "utterance_id": 39,
        }],
    })
    assert set(tools) == {"read_question_context"}
    sources = tools["read_question_context"]("future plans")
    assert [source["utterance_id"] for source in sources] == [39, 40]
    assert all(source["meeting_id"] == "original" for source in sources)


def test_all_materials_mode_keeps_search_tools(tmp_path):
    tools = make_tools(
        tmp_path, question_context={"answer_scope": "all_meetings"}
    )
    assert set(tools) == {
        "search_transcripts", "search_documents", "search_memory"
    }
