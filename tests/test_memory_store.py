import json
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.memory_store import MemoryStore


def _write_facts(root: Path, meeting: str, run: str, items: list[dict]):
    facts = root / "outputs" / meeting / run / "facts" / "facts.json"
    facts.parent.mkdir(parents=True)
    facts.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")


def test_memory_store_versions_decisions_and_keeps_source_attribution():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        _write_facts(root, "m1", "20240101T090000_000000Z", [
            {"id": 1, "kind": "decision", "text": "Choose option A for initial mobile rollout platform.", "source_utterance_ids": [10]},
            {"id": 2, "kind": "risk", "text": "Option A has vendor lock-in risk.", "source_utterance_ids": [11]},
            {"id": 3, "kind": "task", "text": "Review vendor terms.", "owner": "alice", "deadline": "Friday", "source_utterance_ids": [12]},
        ])
        _write_facts(root, "m2", "20240110T090000_000000Z", [
            {"id": 4, "kind": "decision", "text": "Choose option B for the mobile rollout platform after discovering vendor lock-in risk in option A.", "source_utterance_ids": [20]},
            {"id": 5, "kind": "open_question", "text": "Need to confirm migration cost for option B.", "owner": "carol", "source_utterance_ids": [21]},
        ])

        with MemoryStore(root) as memory:
            counts = memory.sync()
            assert counts["decisions"] == 1
            assert counts["decision_versions"] == 2
            assert counts["tasks"] == 1
            assert counts["open_questions"] == 1

            result = memory.reasoning()[0]
            assert "option b" in result["current"]["text"].lower()
            assert result["versions"][0]["status"] == "superseded"
            assert len(result["changes"]) == 1
            related = result["changes"][0]["related_evidence"]
            assert any("vendor lock-in" in item["text"].lower() for item in related)
            assert any(item["source_url"].endswith("utterance/11") for item in related)
            assert "not asserted automatically" in result["changes"][0]["why_changed"]

            question = next(item for item in memory.list_items() if item["kind"] == "open_question")
            task = next(item for item in memory.list_items() if item["kind"] == "task")
            assert question["status"] == "open"
            assert task["owner"] == "alice"
            assert task["deadline"] == "Friday"
            assert memory.update_item_status("open_question", question["question_id"], "resolved")
            memory.sync()
            question_after_sync = next(
                item for item in memory.list_items("open_question")
                if item["question_id"] == question["question_id"]
            )
            assert question_after_sync["status"] == "resolved"


def test_memory_does_not_merge_unrelated_decisions_on_generic_option_words():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        _write_facts(root, "m1", "20240101T090000_000000Z", [
            {"id": 1, "kind": "decision", "text": "Choose option A for the billing database.", "source_utterance_ids": [10]},
        ])
        _write_facts(root, "m2", "20240110T090000_000000Z", [
            {"id": 2, "kind": "decision", "text": "Choose option B for the hiring process.", "source_utterance_ids": [20]},
        ])

        with MemoryStore(root) as memory:
            counts = memory.sync()
            assert counts["decisions"] == 2
            assert counts["decision_versions"] == 2
