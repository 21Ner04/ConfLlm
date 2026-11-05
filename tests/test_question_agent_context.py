from backend.question_agent import run_question_agent


def test_origin_context_reaches_planner():
    states = []
    context = {"meeting_id": "original", "run_id": "r1"}

    def decide(state):
        states.append(state)
        if len(states) == 1:
            return {"action": "search_transcripts", "query": "future plans"}
        return {"action": "finish", "sufficient": False, "claims": []}

    result = run_question_agent(
        "What are the future plans?",
        decide,
        {"search_transcripts": lambda query: []},
        question_context=context,
    )
    assert result["status"] == "insufficient_sources"
    assert states[0]["question_context"] == context


def test_planner_cannot_mutate_original_question_context():
    original = {"meeting_id": "original", "run_id": "r1"}
    calls = []

    def decide(state):
        calls.append(True)
        state["question_context"]["meeting_id"] = "changed"
        if len(calls) == 1:
            return {"action": "search_transcripts", "query": "future plans"}
        return {"action": "finish", "sufficient": False, "claims": []}

    run_question_agent(
        "What are the future plans?",
        decide,
        {"search_transcripts": lambda query: []},
        question_context=original,
    )
    assert original["meeting_id"] == "original"
