import json
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.rag_index import LocalVectorIndex
from backend.memory_store import MemoryStore


def _embed(text: str):
    lowered = text.lower()
    vector = [0.0, 0.0, 0.0]
    if "option a" in lowered or "option b" in lowered or "variant a" in lowered:
        vector[0] = 1.0
    if "vendor" in lowered or "lock" in lowered or "risk" in lowered:
        vector[1] = 1.0
    if "chosen" in lowered or "decision" in lowered or "why" in lowered:
        vector[2] = 1.0
    return vector


def test_rag_index_finds_related_protocol_summary_and_risk():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        meeting_dir = root / "outputs" / "m1" / "run1"
        transcript_dir = meeting_dir / "transcript"
        facts_dir = meeting_dir / "facts"
        transcript_dir.mkdir(parents=True)
        facts_dir.mkdir(parents=True)
        (facts_dir.parent / "pipeline_status.json").write_text(
            json.dumps({"status": "completed_needs_review"}),
            encoding="utf-8",
        )

        transcript = {
            "utterances": [
                {
                    "id": 1,
                    "speaker": "speaker_1",
                    "start": 0,
                    "end": 12,
                    "text": "We decided to choose option A because it is cheaper.",
                },
                {
                    "id": 2,
                    "speaker": "speaker_2",
                    "start": 12,
                    "end": 25,
                    "text": "Later we found option A has a vendor lock-in risk.",
                },
            ]
        }
        facts = {
            "items": [
                {
                    "id": 101,
                    "kind": "decision",
                    "text": "Choose option A for the initial rollout.",
                    "owner": "speaker_1",
                    "source_utterance_ids": [1],
                },
                {
                    "id": 102,
                    "kind": "risk",
                    "text": "Option A has vendor lock-in risk.",
                    "owner": "speaker_2",
                    "source_utterance_ids": [2],
                },
            ]
        }

        (transcript_dir / "aligned_transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
        )
        (facts_dir / "facts.json").write_text(
            json.dumps(facts, ensure_ascii=False), encoding="utf-8"
        )

        with LocalVectorIndex(
            root, embedding_fn=_embed, chunking_fn=lambda text: [text]
        ) as index:
            index.reindex()

            results = index.query(
                "why was the option A chosen and what risk appeared later?",
                top_k=3,
            )

            assert len(results) >= 2
            assert any("option A" in item["text"] or "Choose option A" in item["text"] for item in results)
            assert any("vendor lock-in" in item["text"] or "risk" in item["text"].lower() for item in results)
            assert all(item["meeting_id"] == "m1" for item in results)

            decisions = index.query(
                "option A",
                top_k=5,
                meeting_id="m1",
                topic="initial rollout",
                decision_id="101",
                source_types=["decision"],
            )
            assert len(decisions) == 1
            assert decisions[0]["kind"] == "decision"
            assert decisions[0]["source_item_id"] == 101
            assert decisions[0]["source_url"].endswith("fact/101")

            with MemoryStore(root) as memory:
                stable_decision_id = memory.list_decisions()[0]["decision_id"]
            linked = index.query(
                "option A",
                top_k=5,
                decision_id=stable_decision_id,
                source_types=["decision", "risk"],
            )
            assert {item["source_type"] for item in linked} == {"decision", "risk"}
            assert all(item["decision_id"] == stable_decision_id for item in linked)


def test_rag_prefers_saved_review_revision_and_keeps_evidence_ids():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = root / "outputs" / "m1" / "run1"
        transcript_dir = run_dir / "transcript"
        facts_dir = run_dir / "facts"
        transcript_dir.mkdir(parents=True)
        facts_dir.mkdir(parents=True)
        (facts_dir.parent / "pipeline_status.json").write_text(
            json.dumps({"status": "completed_needs_review"}),
            encoding="utf-8",
        )
        (transcript_dir / "aligned_transcript.json").write_text(
            json.dumps({"utterances": [{"id": 7, "speaker": "speaker_1", "start": 1, "end": 2, "text": "The reviewed platform is option B."}]}),
            encoding="utf-8",
        )
        (facts_dir / "facts.json").write_text(json.dumps({
            "items": [{"id": 1, "kind": "decision", "text": "Choose option A.", "source_utterance_ids": [7]}]
        }), encoding="utf-8")
        (run_dir / "reviewed_protocol.json").write_text(json.dumps({
            "revision": 2,
            "items": [{"id": 1, "kind": "decision", "text": "Choose option B after testing the mobile platform.", "evidence": [{"utterance_id": 7, "quote": "The reviewed platform is option B."}]}],
        }), encoding="utf-8")

        with LocalVectorIndex(
            root, embedding_fn=_embed, chunking_fn=lambda text: [text]
        ) as index:
            index.reindex()
            results = index.query("option B", top_k=5, source_types=["decision"])

        assert len(results) == 1
        assert "option B" in results[0]["text"]
        assert results[0]["review_revision"] == 2
        assert results[0]["source_utterance_ids"] == [7]
        assert results[0]["source_url"].endswith("reviewed_fact/1")
