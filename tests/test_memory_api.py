import json
import importlib.util
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

# job_store currently imports the Linux-only fcntl module at import time.
if "fcntl" not in sys.modules:
    if importlib.util.find_spec("fcntl") is None:
        sys.modules["fcntl"] = types.ModuleType("fcntl")

from backend import main


def test_memory_api_exposes_reasoning_and_resolvable_questions(monkeypatch):
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        facts = root / "outputs" / "meeting-a" / "20240101T090000_000000Z" / "facts" / "facts.json"
        facts.parent.mkdir(parents=True)
        facts.write_text(json.dumps({
            "items": [
                {"id": 1, "kind": "decision", "text": "Choose option A for rollout.", "source_utterance_ids": [10]},
                {"id": 2, "kind": "open_question", "text": "Confirm migration cost for option A.", "owner": "alice", "source_utterance_ids": [11]},
            ]
        }, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(main, "ROOT", root)

        with TestClient(main.app) as client:
            sync = client.post("/api/memory/sync")
            assert sync.status_code == 200
            assert sync.json()["decisions"] == 1

            reasoning = client.get("/api/memory/reasoning", params={"topic": "option A"})
            assert reasoning.status_code == 200
            assert reasoning.json()["count"] == 1
            assert reasoning.json()["decisions"][0]["current"]["text"] == "Choose option A for rollout."

            items = client.get("/api/memory/items", params={"kind": "open_question", "status": "open"})
            question_id = items.json()["items"][0]["question_id"]
            updated = client.patch(
                f"/api/memory/items/open_question/{question_id}",
                json={"status": "resolved"},
            )
            assert updated.status_code == 200
            assert updated.json()["status"] == "resolved"


def test_video_endpoint_supports_browser_range_seeking_and_rejects_external_paths(monkeypatch):
    with TemporaryDirectory() as temporary, TemporaryDirectory() as outside:
        root = Path(temporary)
        meeting_id = "web_" + "a" * 32
        run_id = "20240101T090000_000000Z"
        video = root / "web_jobs" / ("a" * 32) / "source.mp4"
        video.parent.mkdir(parents=True)
        video_bytes = b"0123456789abcdef"
        video.write_bytes(video_bytes)

        run_folder = root / "outputs" / meeting_id / run_id
        run_folder.mkdir(parents=True)
        status_path = run_folder / "pipeline_status.json"
        status_path.write_text(
            json.dumps({"meeting_id": meeting_id, "run_id": run_id, "video": str(video)}),
            encoding="utf-8",
        )
        monkeypatch.setattr(main, "ROOT", root)
        monkeypatch.setattr(main, "OUTPUTS", root / "outputs")

        with TestClient(main.app) as client:
            response = client.get(
                f"/api/meetings/{meeting_id}/{run_id}/video",
                headers={"Range": "bytes=2-5"},
            )
            assert response.status_code == 206
            assert response.content == video_bytes[2:6]
            assert response.headers["content-range"] == f"bytes 2-5/{len(video_bytes)}"
            assert response.headers["content-type"].startswith("video/mp4")

            external_path = Path(outside) / "private.mp4"
            external_path.write_bytes(b"not served")
            status_path.write_text(
                json.dumps({"meeting_id": meeting_id, "run_id": run_id, "video": str(external_path)}),
                encoding="utf-8",
            )
            rejected = client.get(f"/api/meetings/{meeting_id}/{run_id}/video")
            assert rejected.status_code == 404
