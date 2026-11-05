# ConfLlm: separate embeddings and transactional reindex
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]


def _safe_float_list(raw):
    if isinstance(raw, str):
        try:
            return [float(value) for value in json.loads(raw)]
        except json.JSONDecodeError:
            return []
    if isinstance(raw, (list, tuple)):
        try:
            return [float(value) for value in raw]
        except (TypeError, ValueError):
            return []
    return []


class LocalVectorIndex:
    def __init__(
        self,
        root: Path | None = None,
        store_path: Path | None = None,
        embedding_fn: Callable[[str], Sequence[float]] | None = None,
        chunking_fn: Callable[[str], Sequence[str]] | None = None,
    ):
        self.root = Path(root) if root is not None else ROOT
        self.store_path = (
            Path(store_path)
            if store_path is not None
            else self.root / "data" / "rag_index.sqlite3"
        )
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_fn = embedding_fn or self._default_embedding
        # ConfLlm: injectable chunking v1
        self.chunking_fn = chunking_fn
        self._conn = sqlite3.connect(self.store_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _ensure_schema(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                vector TEXT NOT NULL,
                source_type TEXT NOT NULL,
                meeting_id TEXT,
                run_id TEXT,
                speaker TEXT,
                utterance_id TEXT,
                document_id TEXT,
                decision_id TEXT,
                task_id TEXT,
                source_url TEXT,
                timestamp REAL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def clear(self):
        self._conn.execute("DELETE FROM rag_chunks")
        self._conn.commit()

    def _default_embedding(self, text: str):
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError(
                "OpenAI client is not available; configure LLM_BASE_URL, LLM_API_KEY and LLM_MODEL or pass embedding_fn explicitly."
            ) from exc

        base_url = (os.environ.get("LLM_EMBEDDING_BASE_URL") or os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")
        api_key_value = (os.environ.get("LLM_EMBEDDING_API_KEY") or os.environ.get("LLM_API_KEY") or "").strip()
        model = (
            os.environ.get("LLM_EMBEDDING_MODEL")
            or os.environ.get("LLM_MODEL", "")
        ).strip()

        if not base_url or not api_key_value or not model:
            raise RuntimeError(
                "LLM_BASE_URL and LLM_API_KEY must be configured, and the server must expose embeddings for LLM_EMBEDDING_MODEL (or LLM_MODEL)."
            )

        client = OpenAI(
            base_url=base_url,
            api_key=api_key_value,
            timeout=30.0,
            max_retries=0,
        )
        response = client.embeddings.create(model=model, input=text, encoding_format="float")
        return response.data[0].embedding

    @staticmethod
    def _normalize_text(value: str):
        return (value or "").strip()

    # ConfLlm: remote token-aware chunking v1
    def _make_chunks(self, text: str, chunk_size: int = 600):
        if not text.strip():
            return []

        if self.chunking_fn is not None:
            raw = self.chunking_fn(text)
            if isinstance(raw, (str, bytes)):
                raise ValueError("chunking_fn must return a sequence of texts")
            chunks = list(raw)
            if not chunks or any(
                not isinstance(chunk, str) or not chunk.strip()
                for chunk in chunks
            ):
                raise ValueError("chunking_fn returned invalid chunks")
            return chunks

        from urllib.request import Request, urlopen

        if not text.strip():
            return []

        url = os.environ.get("LLM_CHUNKING_URL", "").strip()
        if not url:
            raise RuntimeError(
                "Configure LLM_CHUNKING_URL for token-aware indexing"
            )

        request = Request(
            url,
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=120) as response:
            payload = json.load(response)

        limit = payload.get("max_seq_length")
        entries = payload.get("chunks")

        if not isinstance(limit, int) or not isinstance(entries, list):
            raise RuntimeError("Invalid chunking response")

        result = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError("Invalid chunk entry")
            fragment = entry.get("text")
            count = entry.get("token_count")
            if (
                not isinstance(fragment, str)
                or not fragment.strip()
                or not isinstance(count, int)
                or count <= 0
                or count > limit
            ):
                raise RuntimeError("Invalid or oversized chunk")
            result.append(fragment)

        if not result:
            raise RuntimeError("Chunker returned no chunks for nonempty text")

        return result

    @staticmethod
    def _cosine_similarity(left, right):
        left = [float(value) for value in left]
        right = [float(value) for value in right]
        if len(left) != len(right):
            return 0.0
        if not left:
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        norm_left = math.sqrt(sum(a * a for a in left))
        norm_right = math.sqrt(sum(b * b for b in right))
        if norm_left == 0 or norm_right == 0:
            return 0.0
        return numerator / (norm_left * norm_right)

    def _insert_chunk(self, *, text: str, source_type: str, metadata: dict):
        candidate = self._normalize_text(text)
        if not candidate:
            return
        vector = self.embedding_fn(candidate)
        vector_json = json.dumps([float(value) for value in vector], ensure_ascii=False)
        self._conn.execute(
            """
            INSERT INTO rag_chunks (
                text,
                vector,
                source_type,
                meeting_id,
                run_id,
                speaker,
                utterance_id,
                document_id,
                decision_id,
                task_id,
                source_url,
                timestamp,
                created_at,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate,
                vector_json,
                source_type,
                metadata.get("meeting_id"),
                metadata.get("run_id"),
                metadata.get("speaker"),
                str(metadata.get("utterance_id")) if metadata.get("utterance_id") is not None else None,
                metadata.get("document_id"),
                str(metadata.get("decision_id")) if metadata.get("decision_id") is not None else None,
                str(metadata.get("task_id")) if metadata.get("task_id") is not None else None,
                metadata.get("source_url"),
                metadata.get("timestamp"),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(metadata, ensure_ascii=False),
            ),
        )

    def _index_transcript(self, transcript_path: Path, meeting_id: str, run_id: str):
        if not transcript_path.is_file():
            return
        try:
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return

        utterances = transcript.get("utterances", [])
        for utterance in utterances:
            text = self._normalize_text(utterance.get("text", ""))
            if not text:
                continue
            metadata = {
                "meeting_id": meeting_id,
                "run_id": run_id,
                "speaker": utterance.get("speaker"),
                "utterance_id": utterance.get("id"),
                "start": utterance.get("start"),
                "end": utterance.get("end"),
                "timestamp": utterance.get("start"),
                "source_type": "utterance",
                "source_url": f"meeting://{meeting_id}/{run_id}/utterance/{utterance.get('id')}",
            }
            for chunk in self._make_chunks(text):
                self._insert_chunk(text=chunk, source_type="utterance", metadata=metadata)

    def _index_facts(
        self,
        facts_path: Path,
        meeting_id: str,
        run_id: str,
        decision_links: dict[tuple[str, str, str], str] | None = None,
    ):
        if not facts_path.is_file():
            return
        try:
            facts = json.loads(facts_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return

        self._index_items(facts.get("items", []), meeting_id, run_id, decision_links)

    def _index_items(
        self,
        items: list[dict],
        meeting_id: str,
        run_id: str,
        decision_links: dict[tuple[str, str, str], str] | None = None,
    ):
        for item in items:
            if not isinstance(item, dict):
                continue
            text = self._normalize_text(item.get("text", ""))
            if not text:
                continue
            kind = item.get("kind") or "fact"
            source_item_id = str(item.get("id") if item.get("id") is not None else "")
            stable_decision_id = (decision_links or {}).get(
                (meeting_id, run_id, source_item_id)
            )
            metadata = {
                "meeting_id": meeting_id,
                "run_id": run_id,
                "kind": kind,
                "topic": text,
                "source_item_id": item.get("id"),
                "review_revision": item.get("review_revision"),
                "decision_id": stable_decision_id or item.get("decision_id") or (item.get("id") if kind == "decision" else None),
                "source_decision_id": str(item.get("id")) if kind == "decision" else None,
                "task_id": item.get("id") if item.get("kind") == "task" else None,
                "speaker": item.get("owner") or None,
                "source_utterance_ids": item.get("source_utterance_ids") or [
                    evidence.get("utterance_id")
                    for evidence in item.get("evidence", [])
                    if evidence.get("utterance_id") is not None
                ],
                "source_url": (
                    f"meeting://{meeting_id}/{run_id}/reviewed_fact/{item.get('id')}"
                    if item.get("review_revision") is not None
                    else f"meeting://{meeting_id}/{run_id}/fact/{item.get('id')}"
                ),
                "timestamp": None,
            }
            for chunk in self._make_chunks(text):
                self._insert_chunk(text=chunk, source_type=kind, metadata=metadata)

    def _index_documents(self, base_dir: Path):
        for file_path in sorted(base_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".md", ".txt", ".json"}:
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            metadata = {
                "document_id": file_path.stem,
                "source_type": "document",
                "source_url": str(file_path.resolve()),
            }
            for chunk in self._make_chunks(content):
                self._insert_chunk(text=chunk, source_type="document", metadata=metadata)

    def reindex(self):
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute("DELETE FROM rag_chunks")

            from .memory_store import MemoryStore

            with MemoryStore(self.root) as memory:
                memory.sync()
                decision_links = memory.decision_links()

            outputs_root = self.root / "outputs"
            if outputs_root.exists():
                for meeting_dir in sorted(outputs_root.glob("*")):
                    if not meeting_dir.is_dir():
                        continue
                    for run_dir in sorted(meeting_dir.glob("*")):
                        if not run_dir.is_dir():
                            continue
                        # ConfLlm: index completed runs only
                        status_path = run_dir / "pipeline_status.json"
                        try:
                            run_status = json.loads(status_path.read_text(encoding="utf-8"))
                        except (OSError, ValueError, TypeError):
                            continue
                        if not isinstance(run_status, dict):
                            continue
                        if run_status.get("status") not in {"completed", "completed_needs_review"}:
                            continue
                        
                        transcript_path = run_dir / "transcript" / "aligned_transcript.json"
                        facts_path = run_dir / "facts" / "facts.json"
                        reviewed_path = run_dir / "reviewed_protocol.json"
                        self._index_transcript(transcript_path, meeting_dir.name, run_dir.name)
                        review = {}
                        if reviewed_path.is_file():
                            try:
                                review = json.loads(reviewed_path.read_text(encoding="utf-8"))
                            except (OSError, ValueError, TypeError):
                                review = {}
                        if review.get("revision", 0) > 0 and isinstance(review.get("items"), list):
                            reviewed_items = [
                                {**item, "review_revision": review["revision"]}
                                for item in review["items"] if isinstance(item, dict)
                            ]
                            self._index_items(
                                reviewed_items,
                                meeting_dir.name,
                                run_dir.name,
                                decision_links,
                            )
                        else:
                            self._index_facts(facts_path, meeting_dir.name, run_dir.name, decision_links)

            for base_dir in [self.root / "knowledge", self.root / "docs"]:
                if base_dir.exists():
                    self._index_documents(base_dir)

            self._conn.commit()
            return self.count()

        except BaseException:
            self._conn.rollback()
            raise

    def count(self):
        row = self._conn.execute("SELECT COUNT(*) AS total FROM rag_chunks").fetchone()
        return int(row["total"] if row is not None else 0)

    def query(
        self,
        question: str,
        top_k: int = 5,
        meeting_ids: Sequence[str] | None = None,
        *,
        meeting_id: str | None = None,
        topic: str | None = None,
        decision_id: str | None = None,
        source_types: Sequence[str] | None = None,
    ):
        text = self._normalize_text(question)
        if not text:
            return []
        query_vector = self.embedding_fn(text)

        filters = []
        values = []
        selected_meetings = list(meeting_ids or [])
        if meeting_id:
            selected_meetings.append(meeting_id)
        if selected_meetings:
            placeholders = ", ".join("?" for _ in selected_meetings)
            filters.append(f"meeting_id IN ({placeholders})")
            values.extend(selected_meetings)
        if source_types:
            placeholders = ", ".join("?" for _ in source_types)
            filters.append(f"source_type IN ({placeholders})")
            values.extend(source_types)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        rows = self._conn.execute("SELECT * FROM rag_chunks" + where, values).fetchall()

        scored = []
        for row in rows:
            vector = _safe_float_list(row["vector"])
            score = self._cosine_similarity(query_vector, vector)
            if score <= 0:
                continue
            payload = json.loads(row["metadata_json"])
            payload.update(
                {
                    "text": row["text"],
                    "score": round(score, 6),
                    "source_type": row["source_type"],
                    "meeting_id": row["meeting_id"],
                    "run_id": row["run_id"],
                    "speaker": row["speaker"],
                    "utterance_id": row["utterance_id"],
                    "document_id": row["document_id"],
                    "decision_id": row["decision_id"],
                    "task_id": row["task_id"],
                    "source_url": row["source_url"],
                    "timestamp": row["timestamp"],
                }
            )
            if decision_id and str(payload.get("decision_id")) != str(decision_id) and str(payload.get("source_decision_id")) != str(decision_id):
                continue
            if topic and topic.casefold() not in str(payload.get("topic") or payload.get("text") or "").casefold():
                continue
            scored.append(payload)

        scored.sort(key=lambda entry: entry.get("score", 0.0), reverse=True)
        return scored[:max(1, top_k)]


if __name__ == "__main__":
    index = LocalVectorIndex(ROOT)
    print(index.reindex())
