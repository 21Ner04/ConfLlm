"""Bounded agent loop with injected planner and read-only search tools."""

import hashlib
import json
import time


SEARCH_ACTIONS = {
    "read_question_context",
    "search_transcripts",
    "search_documents",
    "search_memory",
}

SOURCE_FIELDS = (
    "text", "meeting_id", "run_id", "utterance_id",
    "start", "end", "speaker", "source_type",
    "source_url", "document_path", "decision_id", "source_item_id",
)


def run_question_agent(
    question,
    decide,
    tools,
    *,
    question_context=None,
    max_steps=5,
    max_tool_calls=3,
    max_seconds=180,
    clock=time.monotonic,
):
    """Run one question without changing memory or resolving the question.

    decide receives a JSON-compatible state and returns an action dict.
    Each tool accepts a query and returns a list of source dictionaries.
    Time is checked between calls; callbacks need their own timeouts.
    """
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1500:
        raise ValueError("Invalid question")
    if type(max_steps) is not int or not 1 <= max_steps <= 10:
        raise ValueError("Invalid step limit")
    if type(max_tool_calls) is not int or not 1 <= max_tool_calls <= 5:
        raise ValueError("Invalid tool-call limit")
    if isinstance(max_seconds, bool) or not 0 < max_seconds <= 600:
        raise ValueError("Invalid time limit")

    started = clock()
    evidence = []
    evidence_keys = {}
    trace = []
    searched = set()
    tool_calls = 0
    finish_only = False

    def finish(status, reason, answer="", sources=None, claims=None):
        return {
            "status": status,
            "stop_reason": reason,
            "question": question.strip(),
            "answer": answer,
            "claims": claims or [],
            "sources": sources or [],
            "retrieved_sources": evidence,
            "agent_trace": trace,
            "tool_calls": tool_calls,
            "elapsed_ms": round((clock() - started) * 1000, 1),
        }

    for step in range(1, max_steps + 1):
        if clock() - started >= max_seconds:
            return finish("incomplete", "time_limit")

        state = {
            "question": question.strip(),
            "question_context": question_context or {},
            "evidence": evidence,
            "trace": trace,
            "available_tools": [] if finish_only else sorted(
                action for action in SEARCH_ACTIONS
                if callable(tools.get(action))
            ),
            "remaining_tool_calls": (
                0 if finish_only else max_tool_calls - tool_calls
            ),
            "required_action": "finish" if finish_only else None,
            "remaining_steps": max_steps - step,
        }

        try:
            # Prevent callbacks from mutating the agent's internal state.
            decision = decide(json.loads(json.dumps(state, ensure_ascii=False)))
        except Exception:
            return finish("error", "planner_failed")

        if clock() - started >= max_seconds:
            return finish("incomplete", "time_limit")

        if not isinstance(decision, dict):
            return finish("error", "invalid_decision")

        action = decision.get("action")

        if finish_only and action != "finish":
            return finish("incomplete", "final_decision_required")

        if action == "finish":
            if set(decision) != {"action", "sufficient", "claims"}:
                return finish("error", "invalid_finish")

            sufficient = decision["sufficient"]
            claims = decision["claims"]

            if type(sufficient) is not bool or not isinstance(claims, list):
                return finish("error", "invalid_finish")

            if not sufficient:
                if claims:
                    return finish("error", "invalid_finish")
                # A refusal without any successful search is not a finding.
                if not any(item.get("status") == "ok" for item in trace):
                    return finish("incomplete", "no_successful_search")
                transcript_search_available = callable(
                    tools.get("search_transcripts")
                )
                transcript_search_succeeded = any(
                    item.get("action") == "search_transcripts"
                    and item.get("status") == "ok"
                    for item in trace
                )
                if (
                    transcript_search_available
                    and not transcript_search_succeeded
                ):
                    trace.append({
                        "step": step,
                        "action": "finish",
                        "status": "rejected",
                        "reason": (
                            "Search transcripts before concluding "
                            "that evidence is insufficient."
                        ),
                    })
                    continue

                trace.append({"step": step, "action": "finish", "status": "ok"})
                return finish(
                    "insufficient_sources",
                    "model_finished",
                    "Недостаточно оснований для ответа в проверенных источниках.",
                )

            if not 1 <= len(claims) <= 8:
                return finish("error", "invalid_claims")

            allowed = {item["id"] for item in evidence}
            cited = set()
            paragraphs = []

            for claim in claims:
                if not isinstance(claim, dict):
                    return finish("error", "invalid_claims")
                if set(claim) != {"text", "source_ids"}:
                    return finish("error", "invalid_claims")

                text = claim["text"]
                identifiers = claim["source_ids"]

                if (
                    not isinstance(text, str)
                    or not 1 <= len(text.strip()) <= 4000
                    or not isinstance(identifiers, list)
                    or not 1 <= len(identifiers) <= 18
                    or any(type(value) is not int for value in identifiers)
                ):
                    return finish("error", "invalid_claims")

                if not set(identifiers).issubset(allowed):
                    return finish("error", "unknown_source_ids")

                cited.update(identifiers)
                paragraphs.append(
                    text.strip() + " " +
                    "".join(f"[{value}]" for value in sorted(set(identifiers)))
                )

            trace.append({"step": step, "action": "finish", "status": "ok"})
            return finish(
                "answered",
                "model_finished",
                "\n\n".join(paragraphs),
                [item for item in evidence if item["id"] in cited],
                claims=claims,
            )

        if action not in SEARCH_ACTIONS:
            return finish("error", "unknown_action")
        if set(decision) != {"action", "query"}:
            return finish("error", "invalid_search")

        query = decision.get("query")
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 1500:
            return finish("error", "invalid_query")

        query = query.strip()
        signature = (action, " ".join(query.split()).casefold())

        if signature in searched:
            trace.append({
                "step": step,
                "action": action,
                "query": query,
                "status": "duplicate_skipped",
                "reason": (
                    "This search was already executed. No new call was made. "
                    "Finish using the existing evidence; do not search again."
                ),
            })
            finish_only = True
            continue
        if tool_calls >= max_tool_calls:
            return finish("incomplete", "tool_limit")
        if not callable(tools.get(action)):
            return finish("error", "unavailable_tool")

        searched.add(signature)
        tool_calls += 1

        try:
            raw_sources = tools[action](query)
            if not isinstance(raw_sources, list):
                raise ValueError("Tool must return a list")

            # Validate and normalize before adding any sources.
            normalized = []
            for raw in raw_sources[:6]:
                if not isinstance(raw, dict):
                    raise ValueError("Invalid source")
                text = raw.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Source requires text")

                source = {
                    key: raw[key] for key in SOURCE_FIELDS if key in raw
                }
                source["text"] = text.strip()[:3000]
                source["source_type"] = source.get("source_type") or action
                if len(text.strip()) > 3000:
                    source["truncated"] = True

                canonical = json.dumps(
                    source, ensure_ascii=False, sort_keys=True,
                    allow_nan=False,
                )
                key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                normalized.append((key, source))

            added = 0
            for key, source in normalized:
                if key in evidence_keys or len(evidence) >= 18:
                    continue
                source["id"] = len(evidence) + 1
                evidence_keys[key] = source["id"]
                evidence.append(source)
                added += 1

            trace.append({
                "step": step,
                "action": action,
                "query": query,
                "status": "ok",
                "returned": len(raw_sources),
                "added": added,
            })
        except Exception:
            # Do not copy private exception messages into model context.
            trace.append({
                "step": step,
                "action": action,
                "query": query,
                "status": "tool_failed",
            })

    return finish("incomplete", "step_limit")
