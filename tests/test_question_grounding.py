from backend.question_discovery import question_has_literal_support
from backend.question_agent import run_question_agent


def test_greeting_does_not_support_budget_question():
    assert not question_has_literal_support(
        "Какой бюджет согласован для проекта?",
        [{"text": "Леша, привет!"}],
    )


def test_question_must_be_present_not_just_related():
    assert not question_has_literal_support(
        "Какой бюджет согласован для проекта?",
        [{"text": "Обсудили бюджет проекта."}],
    )


def test_literal_question_can_be_extracted_from_longer_passage():
    assert question_has_literal_support(
        "Где следить за трендами?",
        [{"text": "И ещё: где следить за трендами, если они меняются?"}],
    )


def test_memory_only_refusal_is_not_final_when_transcripts_are_available():
    actions = iter([
        {"action": "search_memory", "query": "future plans"},
        {"action": "finish", "sufficient": False, "claims": []},
        {"action": "search_transcripts", "query": "future plans"},
        {
            "action": "finish",
            "sufficient": True,
            "claims": [{"text": "Continue the current work.", "source_ids": [1]}],
        },
    ])

    result = run_question_agent(
        "What are the speaker's future plans?",
        lambda state: next(actions),
        {
            "search_memory": lambda query: [],
            "search_transcripts": lambda query: [{
                "text": "I want to continue the current work.",
                "meeting_id": "m1",
                "run_id": "r1",
                "utterance_id": "40",
            }],
        },
    )

    assert result["status"] == "answered"
    assert result["tool_calls"] == 2
    assert any(
        step.get("action") == "finish"
        and step.get("status") == "rejected"
        for step in result["agent_trace"]
    )


def test_repeated_early_refusal_exhausts_steps_not_evidence():
    actions = iter([
        {"action": "search_memory", "query": "future plans"},
        {"action": "finish", "sufficient": False, "claims": []},
        {"action": "finish", "sufficient": False, "claims": []},
    ])
    result = run_question_agent(
        "What are the future plans?",
        lambda state: next(actions),
        {
            "search_memory": lambda query: [],
            "search_transcripts": lambda query: [],
        },
        max_steps=3,
    )
    assert result["status"] == "incomplete"
    assert result["stop_reason"] == "step_limit"
