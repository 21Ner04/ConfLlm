import json
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.decision_memory import build_decision_timeline


def test_build_decision_timeline_tracks_state_change_over_time():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        outputs = root / "outputs"

        meeting1 = outputs / "m1"
        run1 = meeting1 / "20240101T090000_000000Z"
        run1.mkdir(parents=True)
        (run1 / "facts").mkdir()
        (run1 / "facts" / "facts.json").write_text(
            json.dumps({
                "items": [
                    {"id": 1, "kind": "decision", "text": "Choose option A for initial mobile rollout platform.", "owner": "alice", "source_utterance_ids": [10]},
                    {"id": 2, "kind": "risk", "text": "Option A has vendor lock-in risk.", "owner": "bob", "source_utterance_ids": [11]},
                ]
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        meeting2 = outputs / "m2"
        run2 = meeting2 / "20240110T090000_000000Z"
        run2.mkdir(parents=True)
        (run2 / "facts").mkdir()
        (run2 / "facts" / "facts.json").write_text(
            json.dumps({
                "items": [
                    {"id": 3, "kind": "decision", "text": "Choose option B for the mobile rollout platform after discovering vendor lock-in risk in option A.", "owner": "alice", "source_utterance_ids": [20]},
                    {"id": 4, "kind": "open_question", "text": "Need to confirm migration cost for option B.", "owner": "carol", "source_utterance_ids": [21]},
                ]
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        timeline = build_decision_timeline(root)

        assert len(timeline["groups"]) >= 1
        latest = timeline["current_decisions"][0]
        assert "option b" in latest["text"].lower()
        assert latest["status"] == "current"
        assert any("vendor lock-in" in entry["text"].lower() for entry in timeline["evidence"]) or any("vendor lock-in" in line.lower() for line in timeline["change_summary"].split())
        assert any(item["kind"] == "open_question" for item in timeline["open_questions"])

def test_build_decision_context_explains_why_decision_changed():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        outputs = root / "outputs"

        meeting1 = outputs / "m1"
        run1 = meeting1 / "20240101T090000_000000Z"
        run1.mkdir(parents=True)
        (run1 / "facts").mkdir()
        (run1 / "facts" / "facts.json").write_text(
            json.dumps({
                "items": [
                    {"id": 1, "kind": "decision", "text": "Choose option A for initial rollout.", "owner": "alice", "source_utterance_ids": [10]},
                    {"id": 2, "kind": "risk", "text": "Option A has vendor lock-in risk.", "owner": "bob", "source_utterance_ids": [11]},
                ]
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        meeting2 = outputs / "m2"
        run2 = meeting2 / "20240110T090000_000000Z"
        run2.mkdir(parents=True)
        (run2 / "facts").mkdir()
        (run2 / "facts" / "facts.json").write_text(
            json.dumps({
                "items": [
                    {"id": 3, "kind": "decision", "text": "Choose option B after discovering vendor lock-in risk in option A.", "owner": "alice", "source_utterance_ids": [20]},
                    {"id": 4, "kind": "open_question", "text": "Need to confirm migration cost for option B.", "owner": "carol", "source_utterance_ids": [21]},
                ]
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        context = build_decision_timeline(root)
        current = context["current_decisions"][0]
        change = context["groups"][0]["change_summary"][0]

        assert "option b" in current["text"].lower()
        assert "vendor lock-in" in change["from"].lower() or "vendor lock-in" in change["to"].lower()
        assert any(item["text"].lower().startswith("need to confirm migration cost") for item in context["open_questions"])


def test_timeline_does_not_group_unrelated_decisions_by_generic_words():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        for meeting, run, item_id, text in [
            ("m1", "20240101T090000_000000Z", 1, "Choose option A for the billing database."),
            ("m2", "20240110T090000_000000Z", 2, "Choose option B for the hiring process."),
        ]:
            facts = root / "outputs" / meeting / run / "facts" / "facts.json"
            facts.parent.mkdir(parents=True)
            facts.write_text(json.dumps({
                "items": [{"id": item_id, "kind": "decision", "text": text, "source_utterance_ids": [item_id]}]
            }), encoding="utf-8")

        timeline = build_decision_timeline(root)
        assert len(timeline["groups"]) == 2
        assert len(timeline["current_decisions"]) == 2