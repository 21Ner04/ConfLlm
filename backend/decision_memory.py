import json
import re
from collections import defaultdict
from pathlib import Path

STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "into", "is",
    "it", "its", "of", "on", "or", "the", "to", "we", "with", "was",
    "were", "this", "that", "then", "after", "before", "why", "what",
    "when", "who", "how", "our", "afterward", "later", "need", "should",
    "be", "because", "due", "as", "chosen", "result", "choose", "option",
    "options", "variant", "variants", "decision", "risk", "risks",
    "решение", "решения", "вариант", "варианты", "выбор", "риск", "риски",
    "план", "подход", "задача",
}


def _read_json(path: Path):
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def _run_sort_key(run_name: str):
    return run_name


def _normalize_text(value):
    return (value or "").strip()


def _topic_tokens(text: str):
    cleaned = re.sub(r"[^a-zа-я0-9]+", " ", (text or "").lower())
    tokens = [token for token in cleaned.split() if token and len(token) > 2 and token not in STOPWORDS]
    labels = {
        f"{match.group(1)}:{match.group(2)}"
        for match in re.finditer(r"\b(option|variant)\s+([a-z0-9]+)\b", (text or "").lower())
    }
    labels.update({
        f"{match.group(1)}:{match.group(2)}"
        for match in re.finditer(r"\b(вариант)\s+([а-яё0-9]+)\b", (text or "").lower())
    })
    return set(tokens[:12]) | labels


def _event_key_for_item(item):
    return _normalize_text(item.get("text") or "")


def build_decision_timeline(root: Path | str | None = None):
    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    outputs_root = root_path / "outputs"
    groups = []
    open_questions = []
    evidence = []
    current_decisions = []

    grouped_events = defaultdict(list)
    timeline_events = []
    if outputs_root.exists():
        for meeting_dir in sorted(outputs_root.iterdir(), key=lambda entry: entry.name):
            if not meeting_dir.is_dir():
                continue
            for run_dir in sorted(meeting_dir.iterdir(), key=lambda entry: _run_sort_key(entry.name)):
                if not run_dir.is_dir():
                    continue
                facts_path = run_dir / "facts" / "facts.json"
                facts = _read_json(facts_path)
                for item in facts.get("items", []):
                    text = _normalize_text(item.get("text"))
                    if not text:
                        continue
                    event = {
                        "id": item.get("id"),
                        "kind": item.get("kind"),
                        "text": text,
                        "owner": item.get("owner"),
                        "deadline": item.get("deadline"),
                        "speaker": item.get("owner"),
                        "meeting_id": meeting_dir.name,
                        "run_id": run_dir.name,
                        "source_utterance_ids": item.get("source_utterance_ids", []),
                        "tokens": _topic_tokens(text),
                    }
                    timeline_events.append(event)
                    if item.get("kind") == "open_question":
                        open_questions.append(event)
                    if item.get("kind") in {"decision", "risk", "open_question", "task"}:
                        evidence.append(event)

    for event in timeline_events:
        candidates = []
        minimum_shared = 2 if event["kind"] == "decision" else 1
        for group in grouped_events.values():
            score = max(
                (len(event["tokens"] & related["tokens"]) for related in group),
                default=0,
            )
            if score >= minimum_shared:
                candidates.append((score, group))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        if candidates and (
            len(candidates) == 1 or candidates[0][0] > candidates[1][0]
        ):
            candidates[0][1].append(event)
        else:
            grouped_events[_event_key_for_item(event)[:80]].append(event)

    for batch in grouped_events.values():
        batch.sort(key=lambda entry: (entry["meeting_id"], entry["run_id"]))
        if not batch:
            continue
        decisions = [entry for entry in batch if entry["kind"] == "decision"]
        if decisions:
            current = decisions[-1]
            current["status"] = "current"
            current_decisions.append(current)
            for prev in decisions[:-1]:
                prev["status"] = "superseded"
        else:
            current = batch[-1]
            current["status"] = "current"
            current_decisions.append(current)

        history = []
        for entry in batch:
            history.append({
                "meeting_id": entry["meeting_id"],
                "run_id": entry["run_id"],
                "id": entry["id"],
                "kind": entry["kind"],
                "text": entry["text"],
                "owner": entry.get("owner"),
                "status": entry.get("status", "historical"),
                "source_utterance_ids": entry.get("source_utterance_ids", []),
            })

        change_summary = []
        previous_decision = None
        for entry in batch:
            if entry["kind"] == "decision":
                if previous_decision is not None and entry["text"] != previous_decision["text"]:
                    change_summary.append({
                        "from": previous_decision["text"],
                        "to": entry["text"],
                        "reason": "Decision changed after reviewing evidence and constraints.",
                        "meeting_id": entry["meeting_id"],
                        "run_id": entry["run_id"],
                    })
                previous_decision = entry

        groups.append({
            "topic": next((entry["text"] for entry in batch if entry["kind"] == "decision"), batch[0]["text"]),
            "timeline": history,
            "current_decision": current,
            "change_summary": change_summary,
        })

    scoped_open_questions = []
    for item in open_questions:
        scoped_open_questions.append({
            "id": item["id"],
            "meeting_id": item["meeting_id"],
            "run_id": item["run_id"],
            "kind": item["kind"],
            "text": item["text"],
            "owner": item.get("owner"),
            "source_utterance_ids": item.get("source_utterance_ids", []),
        })

    return {
        "groups": groups,
        "current_decisions": current_decisions,
        "open_questions": scoped_open_questions,
        "evidence": [
            {
                "meeting_id": item["meeting_id"],
                "run_id": item["run_id"],
                "kind": item["kind"],
                "text": item["text"],
                "owner": item.get("owner"),
                "source_utterance_ids": item.get("source_utterance_ids", []),
            }
            for item in evidence
        ],
        "change_summary": "; ".join(
            f"{detail['from']} -> {detail['to']}"
            for detail in [
                item for group in groups for item in group["change_summary"]
            ]
        ) or "No decision changes detected.",
    }
