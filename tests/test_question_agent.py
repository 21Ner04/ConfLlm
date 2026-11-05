from copy import deepcopy

import pytest

from backend.question_agent import run_question_agent


def planner(*actions):
    iterator = iter(actions)
    return lambda state: deepcopy(next(iterator))


def search(action="search_transcripts", query="pilot date"):
    return {"action": action, "query": query}


def supported(text="20 November", source_ids=None):
    return {
        "action": "finish",
        "sufficient": True,
        "claims": [{
            "text": text,
            "source_ids": [1] if source_ids is None else source_ids,
        }],
    }


def source(text="The pilot starts on 20 November."):
    return {
        "text": text,
        "meeting_id": "m2",
        "run_id": "r2",
        "utterance_id": "10",
        "source_type": "utterance",
    }


def test_agent_searches_then_returns_cited_answer():
    result = run_question_agent(
        "When is the pilot?",
        planner(search(), supported()),
        {"search_transcripts": lambda query: [source()]},
    )
    assert result["status"] == "answered"
    assert result["answer"] == "20 November [1]"
    assert result["sources"][0]["utterance_id"] == "10"
    assert result["tool_calls"] == 1


def test_model_can_choose_another_tool_after_empty_search():
    states = []
    actions = iter([
        search(),
        search("search_documents", "pilot approval"),
        supported(),
    ])

    def decide(state):
        states.append(deepcopy(state))
        return next(actions)

    result = run_question_agent(
        "When is the pilot?",
        decide,
        {
            "search_transcripts": lambda query: [],
            "search_documents": lambda query: [source()],
        },
    )
    assert result["status"] == "answered"
    assert result["tool_calls"] == 2
    assert states[1]["trace"][0]["returned"] == 0
    assert len(states[2]["evidence"]) == 1


def test_unknown_source_identifier_is_rejected():
    result = run_question_agent(
        "When?",
        planner(search(), supported(source_ids=[99])),
        {"search_transcripts": lambda query: [source()]},
    )
    assert result["status"] == "error"
    assert result["stop_reason"] == "unknown_source_ids"
    assert result["sources"] == []


def test_repeated_search_is_not_executed_twice():
    calls = []

    def tool(query):
        calls.append(query)
        return []

    result = run_question_agent(
        "When?",
        planner(
            search(),
            search(),
            {"action": "finish", "sufficient": False, "claims": []},
        ),
        {"search_transcripts": tool},
    )
    assert result["status"] == "insufficient_sources"
    assert len(calls) == 1
    assert result["tool_calls"] == 1
    assert any(
        step["status"] == "duplicate_skipped"
        for step in result["agent_trace"]
    )



def test_tool_budget_does_not_become_insufficient_evidence():
    result = run_question_agent(
        "When?",
        planner(search(), search(query="another query")),
        {"search_transcripts": lambda query: []},
        max_tool_calls=1,
    )
    assert result["status"] == "incomplete"
    assert result["stop_reason"] == "tool_limit"


def test_step_budget_is_reported():
    result = run_question_agent(
        "When?", planner(search()),
        {"search_transcripts": lambda query: []},
        max_steps=1,
    )
    assert result["status"] == "incomplete"
    assert result["stop_reason"] == "step_limit"


def test_failed_tool_does_not_prove_absence_of_answer():
    def broken(query):
        raise RuntimeError("private service details")

    result = run_question_agent(
        "When?",
        planner(
            search(),
            {"action": "finish", "sufficient": False, "claims": []},
        ),
        {"search_transcripts": broken},
    )
    assert result["status"] == "incomplete"
    assert result["stop_reason"] == "no_successful_search"
    assert "private service details" not in str(result)


def test_empty_successful_search_allows_explicit_refusal():
    result = run_question_agent(
        "What budget?",
        planner(
            search(),
            {"action": "finish", "sufficient": False, "claims": []},
        ),
        {"search_transcripts": lambda query: []},
    )
    assert result["status"] == "insufficient_sources"


def test_sources_are_deduplicated_across_queries():
    result = run_question_agent(
        "When?",
        planner(search(), search(query="other wording"), supported()),
        {"search_transcripts": lambda query: [source()]},
    )
    assert result["status"] == "answered"
    assert len(result["retrieved_sources"]) == 1
    assert result["agent_trace"][1]["added"] == 0


def test_time_is_checked_after_planner():
    clock_values = iter([0, 0, 10, 10])
    result = run_question_agent(
        "When?",
        planner(search()),
        {"search_transcripts": lambda query: []},
        max_seconds=1,
        clock=lambda: next(clock_values),
    )
    assert result["status"] == "incomplete"
    assert result["stop_reason"] == "time_limit"
    assert result["tool_calls"] == 0


@pytest.mark.parametrize("limit", [0, 11, True])
def test_invalid_step_limit_is_rejected(limit):
    with pytest.raises(ValueError):
        run_question_agent("When?", lambda state: {}, {}, max_steps=limit)
