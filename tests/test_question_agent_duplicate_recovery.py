from backend.question_agent import run_question_agent


def test_duplicate_recovery_requires_final_action_and_preserves_evidence():
    states = []
    actions = iter([
        {"action": "search_transcripts", "query": "pilot"},
        {"action": "search_transcripts", "query": "pilot"},
        {
            "action": "finish",
            "sufficient": True,
            "claims": [{"text": "20 November", "source_ids": [1]}],
        },
    ])

    def decide(state):
        states.append(state)
        return next(actions)

    result = run_question_agent(
        "When?",
        decide,
        {"search_transcripts": lambda query: [{
            "text": "Launch is on 20 November.",
            "meeting_id": "m1",
            "run_id": "r1",
            "utterance_id": 1,
        }]},
    )
    assert result["status"] == "answered"
    assert result["tool_calls"] == 1
    assert states[2]["required_action"] == "finish"
    assert states[2]["available_tools"] == []
    assert states[2]["remaining_tool_calls"] == 0
    assert len(states[2]["evidence"]) == 1


def test_ignoring_final_action_does_not_become_success():
    actions = iter([
        {"action": "search_transcripts", "query": "pilot"},
        {"action": "search_transcripts", "query": "pilot"},
        {"action": "search_documents", "query": "another query"},
    ])
    calls = []

    def transcripts(query):
        calls.append(query)
        return []

    result = run_question_agent(
        "When?",
        lambda state: next(actions),
        {
            "search_transcripts": transcripts,
            "search_documents": lambda query: [],
        },
    )
    assert result["status"] == "incomplete"
    assert result["stop_reason"] == "final_decision_required"
    assert len(calls) == 1
