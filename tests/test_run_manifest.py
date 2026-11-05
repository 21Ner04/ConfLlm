import json
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.run_manifest import build_run_manifest, record_artifact_schema_fingerprints


def test_manifest_pins_prompt_and_code_and_redacts_endpoint_credentials(monkeypatch):
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "run_pipeline.py").write_text("VERSION = 'pipe_v1'\n", encoding="utf-8")
        (root / "transcribe_parakeet_chunks.py").write_text("MODEL = 'parakeet-test'\n", encoding="utf-8")
        (root / "merge_asr_transcript.py").write_text("SCHEMA_VERSION = 'aligned_v1'\n", encoding="utf-8")
        (root / "resolve_speaker_names.py").write_text("VERSION = 'names_v1'\nPROMPT = 'name prompt'\n", encoding="utf-8")
        (root / "extract_meeting_facts.py").write_text("VERSION = 'facts_v1'\nPROMPT = 'fact prompt'\n", encoding="utf-8")
        video = root / "source.mp4"
        video.write_bytes(b"video")
        monkeypatch.setenv("LLM_MODEL", "qwen-test")
        monkeypatch.setenv("LLM_BASE_URL", "http://secret-user:secret-password@localhost:8001/v1?token=hidden")
        monkeypatch.setenv("LLM_API_KEY", "do-not-store-this")

        manifest = build_run_manifest(
            root=root,
            run_id="run-1",
            meeting_id="meeting-1",
            video_path=video,
            started_at="2025-01-01T00:00:00+00:00",
        )

        assert manifest["models"]["llm"] == "qwen-test"
        assert manifest["prompts"]["fact_extraction"]["version"] == "facts_v1"
        assert manifest["prompts"]["fact_extraction"]["sha256"]
        assert manifest["implementation_sha256"]["pipeline"]
        assert manifest["input"]["video"]["sha256"]
        serialized = json.dumps(manifest)
        assert "secret-password" not in serialized
        assert "do-not-store-this" not in serialized
        assert "token=hidden" not in serialized


def test_manifest_records_exact_runtime_schema_fingerprints():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "outputs"
        facts = output / "facts"
        names = output / "speaker_names"
        facts.mkdir(parents=True)
        names.mkdir(parents=True)
        (facts / "run_config.json").write_text(json.dumps({
            "version": "facts_v3",
            "schema": {"type": "object", "properties": {"items": {"type": "array"}}},
        }), encoding="utf-8")
        (names / "run_config.json").write_text(json.dumps({
            "schema_version": "names_v2",
            "schema": {"type": "object", "properties": {"candidates": {"type": "array"}}},
        }), encoding="utf-8")
        manifest = {}

        record_artifact_schema_fingerprints(manifest, output)

        assert manifest["schema_fingerprints"]["fact_extraction"]["version"] == "facts_v3"
        assert len(manifest["schema_fingerprints"]["fact_extraction"]["sha256"]) == 64
        assert manifest["schema_fingerprints"]["speaker_names"]["version"] == "names_v2"
