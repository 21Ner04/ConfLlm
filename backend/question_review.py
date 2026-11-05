"""Bounded review of explicit open questions; no automatic resolution."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ANSWER_STATUSES = {"answered", "no_sources", "insufficient_sources"}


class QuestionReviewStore:
    def __init__(self, root):
        path = Path(root) / "data" / "question_reviews.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=15)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS question_reviews (
                review_id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
        """)
        self.connection.commit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()

    def save(self, question, result):
        status = result.get("status")
        if status not in ANSWER_STATUSES:
            raise ValueError("Unsupported answer status")

        answer = result.get("answer")
        sources = result.get("sources", [])
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Missing answer")
        if not isinstance(sources, list):
            raise ValueError("Invalid sources")
        if status == "answered" and (
            not sources or not all(isinstance(source, dict) for source in sources)
        ):
            raise ValueError("Candidate answer requires sources")
        if status != "answered" and sources:
            raise ValueError("Unanswered result must not contain cited sources")

        payload = {
            "question_id": question["question_id"],
            "question_text": question["text"],
            "question_meeting_id": question.get("meeting_id"),
            "question_run_id": question.get("run_id"),
            "status": "candidate_answer" if status == "answered" else status,
            "answer": answer,
            "sources": sources,
        }

        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        review_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        payload.update({
            "review_id": review_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "requires_human_review": True,
            "metrics": result.get("metrics"),
        })

        with self.connection:
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO question_reviews VALUES (?, ?, ?, ?)",
                (
                    review_id,
                    question["question_id"],
                    payload["created_at"],
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            created = cursor.rowcount == 1
            row = self.connection.execute(
                "SELECT payload_json FROM question_reviews WHERE review_id=?",
                (review_id,),
            ).fetchone()

        return json.loads(row["payload_json"]), created

    def list_reviews(self, question_id=None, limit=50):
        if not 1 <= limit <= 200:
            raise ValueError("Invalid limit")
        if question_id is None:
            rows = self.connection.execute(
                "SELECT payload_json FROM question_reviews "
                "ORDER BY created_at DESC, review_id LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT payload_json FROM question_reviews WHERE question_id=? "
                "ORDER BY created_at DESC, review_id LIMIT ?",
                (question_id, limit),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]


def review_questions(root, questions, answer_fn, *, limit=1, question_ids=None):
    if not 1 <= limit <= 3:
        raise ValueError("Limit must be between 1 and 3")

    selected_ids = set(question_ids) if question_ids is not None else None
    eligible = [
        question for question in questions
        if question.get("kind") == "open_question"
        and question.get("status") in {"open", "in_progress"}
        and (selected_ids is None or question.get("question_id") in selected_ids)
    ]
    eligible.sort(key=lambda question: (
        str(question.get("meeting_id", "")),
        str(question.get("run_id", "")),
        str(question.get("question_id", "")),
    ))

    results = []
    errors = []

    with QuestionReviewStore(root) as store:
        for question in eligible[:limit]:
            question_id = question.get("question_id")
            text = question.get("text")
            if (
                not isinstance(question_id, str)
                or not question_id
                or not isinstance(text, str)
                or not 1 <= len(text.strip()) <= 1500
            ):
                errors.append({
                    "question_id": question_id,
                    "code": "invalid_question",
                })
                continue

            try:
                result = answer_fn(text.strip())
                saved, created = store.save(question, result)
                results.append({"created": created, "review": saved})
            except Exception as error:
                status_code = getattr(error, "status_code", None)
                entry = {
                    "question_id": question_id,
                    "code": "review_failed",
                }
                if isinstance(status_code, int):
                    entry["http_status"] = status_code
                errors.append(entry)

    attempted = min(limit, len(eligible))
    return {
        "status": "partial" if errors else "ok",
        "eligible": len(eligible),
        "attempted": attempted,
        "remaining": len(eligible) - attempted,
        "created": sum(item["created"] for item in results),
        "results": results,
        "errors": errors,
    }
