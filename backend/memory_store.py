"""Structured, source-attributed meeting memory stored in SQLite."""

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .decision_memory import _topic_tokens

ROOT = Path(__file__).resolve().parents[1]
GENERIC_DECISION_TOKENS = {
    "choose", "chosen", "decision", "option", "options", "variant",
    "variants", "risk", "risks", "approach", "plan", "решение",
    "решения", "вариант", "варианты", "выбор", "риск", "риски",
    "план", "подход",
}


def _stable_id(kind: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{kind}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _event_tokens(item: dict[str, Any]) -> set[str]:
    return _topic_tokens(str(item.get("text") or ""))


def _decision_tokens(item: dict[str, Any]) -> set[str]:
    text = str(item.get("text") or "").casefold()
    tokens = _event_tokens(item) - GENERIC_DECISION_TOKENS
    for match in re.finditer(r"\b(option|variant)\s+([a-z0-9]+)\b", text):
        tokens.add(f"{match.group(1)}:{match.group(2)}")
    for match in re.finditer(r"\b(вариант)\s+([а-яё0-9]+)\b", text):
        tokens.add(f"{match.group(1)}:{match.group(2)}")
    return tokens


def _source_ids(item: dict[str, Any]) -> list[int | str]:
    ids = item.get("source_utterance_ids") or []
    if not ids:
        ids = [e.get("utterance_id") for e in item.get("evidence", []) if e.get("utterance_id") is not None]
    return ids


def _source_url(meeting_id: str, run_id: str, source_item_id: object, utterance_ids: list) -> str:
    if utterance_ids:
        return f"meeting://{meeting_id}/{run_id}/utterance/{utterance_ids[0]}"
    return f"meeting://{meeting_id}/{run_id}/fact/{source_item_id}"


class MemoryStore:
    """Materializes decisions, versions, actions, and evidence from meeting outputs."""

    def __init__(self, root: Path | str | None = None, store_path: Path | None = None):
        self.root = Path(root) if root is not None else ROOT
        self.store_path = Path(store_path) if store_path else self.root / "data" / "meeting_memory.sqlite3"
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.store_path)
        self.connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _ensure_schema(self):
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                current_version_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decision_versions (
                version_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL REFERENCES decisions(decision_id),
                version_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                meeting_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                source_item_id TEXT,
                status TEXT NOT NULL,
                UNIQUE(decision_id, version_number)
            );
            CREATE TABLE IF NOT EXISTS open_questions (
                question_id TEXT PRIMARY KEY,
                decision_id TEXT,
                text TEXT NOT NULL,
                owner TEXT,
                deadline TEXT,
                status TEXT NOT NULL,
                meeting_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                source_item_id TEXT
            );
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                decision_id TEXT,
                text TEXT NOT NULL,
                owner TEXT,
                deadline TEXT,
                status TEXT NOT NULL,
                meeting_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                source_item_id TEXT
            );
            CREATE TABLE IF NOT EXISTS evidence_refs (
                evidence_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                meeting_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                source_item_id TEXT,
                utterance_ids_json TEXT NOT NULL,
                source_url TEXT NOT NULL,
                text TEXT NOT NULL,
                kind TEXT,
                UNIQUE(entity_type, entity_id, meeting_id, run_id, source_item_id)
            );
            CREATE INDEX IF NOT EXISTS idx_versions_decision ON decision_versions(decision_id, version_number);
            CREATE INDEX IF NOT EXISTS idx_evidence_entity ON evidence_refs(entity_type, entity_id);
            CREATE INDEX IF NOT EXISTS idx_questions_status ON open_questions(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            """
        )
        self.connection.commit()

    def _read_run_events(self):
        events = []
        outputs = self.root / "outputs"
        if not outputs.is_dir():
            return events
        for meeting_dir in sorted(outputs.iterdir(), key=lambda p: p.name):
            if not meeting_dir.is_dir():
                continue
            for run_dir in sorted(meeting_dir.iterdir(), key=lambda p: p.name):
                facts = _read_json(run_dir / "facts" / "facts.json")
                items = facts.get("items", [])
                reviewed = _read_json(run_dir / "reviewed_protocol.json")
                if reviewed.get("revision", 0) > 0 and isinstance(reviewed.get("items"), list):
                    items = reviewed["items"]
                for item in items:
                    if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                        continue
                    source_item_id = item.get("id")
                    ids = _source_ids(item)
                    events.append({
                        "id": source_item_id,
                        "kind": item.get("kind"),
                        "text": str(item["text"]).strip(),
                        "owner": item.get("owner"),
                        "deadline": item.get("deadline"),
                        "topic_key": item.get("topic_key") or item.get("decision_key"),
                        "meeting_id": meeting_dir.name,
                        "run_id": run_dir.name,
                        "source_utterance_ids": ids,
                        "source_url": _source_url(meeting_dir.name, run_dir.name, source_item_id, ids),
                        "tokens": _event_tokens(item),
                        "decision_tokens": _decision_tokens(item),
                    })
        return events

    @staticmethod
    def _match_decision(event, groups, minimum_shared: int = 1):
        candidates = []
        for group in groups:
            if event.get("topic_key") and event["topic_key"] == group.get("topic_key"):
                candidates.append((10_000, group))
                continue
            event_tokens = event["decision_tokens"]
            shared_count = max(
                (
                    len(event_tokens & related["decision_tokens"])
                    for related in group["events"]
                ),
                default=0,
            )
            if shared_count >= minimum_shared:
                candidates.append((shared_count, group))
        if not candidates:
            return None
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            return None
        return candidates[0][1]

    def sync(self) -> dict[str, int]:
        events = self._read_run_events()
        events.sort(key=lambda e: (e["run_id"], e["meeting_id"], str(e["id"])))
        groups = []
        for event in events:
            if event["kind"] != "decision":
                continue
            group = self._match_decision(event, groups, minimum_shared=2)
            if group is None:
                group = {"versions": [], "events": [], "topic_key": event.get("topic_key")}
                groups.append(group)
            group["versions"].append(event)
            group["events"].append(event)

        for event in events:
            if event["kind"] == "decision":
                continue
            group = self._match_decision(event, groups, minimum_shared=1)
            if group is not None:
                group["events"].append(event)

        preserved_status = {}
        for table, id_column in (("open_questions", "question_id"), ("tasks", "task_id")):
            try:
                preserved_status[table] = {
                    row[id_column]: row["status"]
                    for row in self.connection.execute(f"SELECT {id_column}, status FROM {table}")
                }
            except sqlite3.OperationalError:
                preserved_status[table] = {}

        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            self.connection.execute("DELETE FROM evidence_refs")
            self.connection.execute("DELETE FROM decision_versions")
            self.connection.execute("DELETE FROM open_questions")
            self.connection.execute("DELETE FROM tasks")
            self.connection.execute("DELETE FROM decisions")

            event_decision: dict[int, str] = {}
            for group in groups:
                first = group["versions"][0]
                decision_id = _stable_id("decision", first["meeting_id"], first["run_id"], first["id"])
                for event in group["events"]:
                    event_decision[id(event)] = decision_id
                versions = group["versions"]
                current_version_id = None
                for index, version in enumerate(versions, 1):
                    version_id = _stable_id("version", version["meeting_id"], version["run_id"], version["id"])
                    current_version_id = version_id
                    status = "current" if index == len(versions) else "superseded"
                    self.connection.execute(
                        "INSERT INTO decision_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (version_id, decision_id, index, version["text"], version["meeting_id"], version["run_id"], str(version["id"]), status),
                    )
                    self._add_evidence("decision_version", version_id, version)
                self.connection.execute(
                    "INSERT INTO decisions VALUES (?, ?, ?, ?, ?)",
                    (decision_id, first["text"], current_version_id, first["meeting_id"], now),
                )
                for event in group["events"]:
                    if event["kind"] != "decision":
                        self._add_evidence("decision", decision_id, event)

            for event in events:
                kind = event["kind"]
                decision_id = event_decision.get(id(event))
                item_id = str(event["id"] if event["id"] is not None else "")
                if kind in {"task", "open_question"}:
                    entity_type = "task" if kind == "task" else "open_question"
                    entity_id = _stable_id(entity_type, event["meeting_id"], event["run_id"], item_id)
                    table = "tasks" if kind == "task" else "open_questions"
                    id_column = "task_id" if kind == "task" else "question_id"
                    status = preserved_status.get(table, {}).get(entity_id, "open")
                    self.connection.execute(
                        f"INSERT INTO {table} ({id_column}, decision_id, text, owner, deadline, status, meeting_id, run_id, source_item_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (entity_id, decision_id, event["text"], event.get("owner"), event.get("deadline"), status, event["meeting_id"], event["run_id"], item_id),
                    )
                    self._add_evidence(entity_type, entity_id, event)
                elif kind not in {"decision"}:
                    entity_id = decision_id or _stable_id("evidence", event["meeting_id"], event["run_id"], item_id)
                    self._add_evidence("decision" if decision_id else "fact", entity_id, event)

        return {
            "decisions": len(groups),
            "decision_versions": len(events_of_kind(events, "decision")),
            "open_questions": len(events_of_kind(events, "open_question")),
            "tasks": len(events_of_kind(events, "task")),
            "evidence_refs": int(self.connection.execute("SELECT COUNT(*) FROM evidence_refs").fetchone()[0]),
        }

    def _add_evidence(self, entity_type: str, entity_id: str, event: dict):
        evidence_id = _stable_id("evidence", entity_type, entity_id, event["meeting_id"], event["run_id"], event["id"])
        self.connection.execute(
            "INSERT OR IGNORE INTO evidence_refs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (evidence_id, entity_type, entity_id, event["meeting_id"], event["run_id"], str(event["id"] if event["id"] is not None else ""), json.dumps(event["source_utterance_ids"], ensure_ascii=False), event["source_url"], event["text"], event["kind"]),
        )

    def list_decisions(self):
        decisions = self.connection.execute(
            "SELECT d.decision_id, d.topic, v.version_id, v.version_number, v.text, v.meeting_id, v.run_id FROM decisions d JOIN decision_versions v ON v.version_id=d.current_version_id ORDER BY v.run_id, v.meeting_id, d.decision_id"
        ).fetchall()
        return [dict(row) for row in decisions]

    def decision_links(self) -> dict[tuple[str, str, str], str]:
        rows = self.connection.execute(
            "SELECT e.meeting_id, e.run_id, e.source_item_id, "
            "CASE WHEN e.entity_type='decision_version' THEN v.decision_id ELSE e.entity_id END AS decision_id "
            "FROM evidence_refs e LEFT JOIN decision_versions v "
            "ON e.entity_type='decision_version' AND e.entity_id=v.version_id "
            "WHERE e.entity_type IN ('decision', 'decision_version')"
        ).fetchall()
        return {
            (row["meeting_id"], row["run_id"], row["source_item_id"]): row["decision_id"]
            for row in rows
            if row["decision_id"]
        }

    def reasoning(self, decision_id: str | None = None, topic: str | None = None):
        rows = self.connection.execute(
            "SELECT decision_id, topic, current_version_id FROM decisions ORDER BY created_at, decision_id"
        ).fetchall()
        results = []
        for decision in rows:
            if decision_id and decision["decision_id"] != decision_id:
                continue
            if topic and topic.casefold() not in decision["topic"].casefold() and not any(topic.casefold() in row["text"].casefold() for row in self.connection.execute("SELECT text FROM decision_versions WHERE decision_id=?", (decision["decision_id"],))):
                continue
            versions = [dict(row) for row in self.connection.execute(
                "SELECT version_id, version_number, text, meeting_id, run_id, status FROM decision_versions WHERE decision_id=? ORDER BY version_number",
                (decision["decision_id"],),
            )]
            changes = []
            for previous, current in zip(versions, versions[1:]):
                if previous["text"].strip() == current["text"].strip():
                    continue
                evidence = [dict(row) for row in self.connection.execute(
                    "SELECT e.meeting_id, e.run_id, e.source_item_id, e.utterance_ids_json, e.source_url, e.text, e.kind FROM evidence_refs e WHERE e.entity_type='decision' AND e.entity_id=? AND e.run_id>=? AND e.run_id<=? ORDER BY e.run_id, e.meeting_id",
                    (decision["decision_id"], previous["run_id"], current["run_id"]),
                )]
                # Evidence text is intentionally sourced from stored facts below, not inferred as causation.
                changes.append({
                    "from": previous,
                    "to": current,
                    "why_changed": "The source record contains related evidence for this version; causal linkage is not asserted automatically." if evidence else "The reason for the change is not recorded in the available evidence.",
                    "related_evidence": evidence,
                })
            current = next((v for v in versions if v["version_id"] == decision["current_version_id"]), None)
            results.append({
                "decision_id": decision["decision_id"],
                "topic": decision["topic"],
                "current": current,
                "versions": versions,
                "changes": changes,
            })
        return results

    def list_items(self, kind: str | None = None, status: str | None = None):
        selections = []
        if kind in (None, "open_question"):
            selections.extend(("open_question", dict(row)) for row in self.connection.execute("SELECT * FROM open_questions" + (" WHERE status=?" if status else "") + " ORDER BY meeting_id, run_id", (status,) if status else ()))
        if kind in (None, "task"):
            selections.extend(("task", dict(row)) for row in self.connection.execute("SELECT * FROM tasks" + (" WHERE status=?" if status else "") + " ORDER BY meeting_id, run_id", (status,) if status else ()))
        return [{"kind": item_kind, **row} for item_kind, row in selections]

    def update_item_status(self, kind: str, item_id: str, status: str) -> bool:
        table, column = ("tasks", "task_id") if kind == "task" else ("open_questions", "question_id")
        cursor = self.connection.execute(f"UPDATE {table} SET status=? WHERE {column}=?", (status, item_id))
        self.connection.commit()
        return cursor.rowcount > 0


def events_of_kind(events: list[dict], kind: str) -> list[dict]:
    return [event for event in events if event["kind"] == kind]


def sync_memory(root: Path | str | None = None) -> dict[str, int]:
    with MemoryStore(root) as store:
        return store.sync()
