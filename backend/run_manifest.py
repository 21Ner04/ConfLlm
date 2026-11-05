"""Build reproducibility manifests without copying credentials or source media."""

import ast
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_constants(path: Path) -> dict[str, Any]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            value_node = node.value
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        result[target.id] = ast.literal_eval(value_node)
                    except (ValueError, TypeError):
                        continue
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                result[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return result


def _safe_endpoint(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value.strip())
        if not parsed.scheme or not parsed.hostname:
            return "configured"
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return "configured"


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_run_manifest(
    *,
    root: Path,
    run_id: str,
    meeting_id: str,
    video_path: Path,
    started_at: str,
    endpoints: dict[str, str | None] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    video_path = video_path.resolve()
    scripts = {
        "pipeline": root / "run_pipeline.py",
        "asr": root / "transcribe_parakeet_chunks.py",
        "alignment": root / "merge_asr_transcript.py",
        "speaker_names": root / "resolve_speaker_names.py",
        "fact_extraction": root / "extract_meeting_facts.py",
        "vad": root / "detect_speech_silero.py",
        "chunking": root / "prepare_asr_chunks.py",
        "export": root / "export_meeting_report.py",
    }
    constants = {key: _module_constants(path) for key, path in scripts.items()}
    prompt_versions = {}
    schema_versions = {}
    for key in ("speaker_names", "fact_extraction"):
        values = constants[key]
        prompt = values.get("PROMPT")
        prompt_versions[key] = {
            "version": values.get("VERSION") or values.get("PROMPT_VERSION") or "unversioned",
            "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest() if isinstance(prompt, str) else None,
        }
        if values.get("VERSION"):
            schema_versions[key] = values["VERSION"]
    schema_versions.update({
        "transcript": "aligned_transcript_v1",
        "review": "reviewed_protocol_v1",
    })
    recorded_config = {
        "audio": {"sample_rate_hz": 16000, "channels": 1, "format": "pcm_s16le"},
        "vad_parameters": constants["vad"].get("parameters"),
        "asr_chunk_max_seconds": 20,
        "asr_model": constants["asr"].get("MODEL"),
        "alignment_max_utterance_gap_seconds": 1.0,
        "speaker_name_attempt_limit": 3,
        "fact_extraction_attempt_limit": 3,
        "fact_extraction_max_block_chars": 4000,
        "llm_temperature": 0,
        "llm_thinking_enabled": False,
    }
    recorded_config.update(config or {})

    try:
        video_stat = video_path.stat()
        video_record = {
            "filename": video_path.name,
            "size_bytes": video_stat.st_size,
            "sha256": _sha256_file(video_path),
        }
    except OSError:
        video_record = {"filename": video_path.name, "available": False}

    return {
        "manifest_schema_version": "run_manifest_v1",
        "meeting_id": meeting_id,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "application_revision": _git_revision(root),
        "python_version": sys.version.split()[0],
        "input": {"video": video_record},
        "models": {
            "llm": os.environ.get("LLM_MODEL"),
            "embedding": os.environ.get("LLM_EMBEDDING_MODEL") or os.environ.get("LLM_MODEL"),
            "parakeet": constants["asr"].get("MODEL"),
            "nemotron": os.environ.get("NEMOTRON_MODEL") or "not reported by endpoint",
        },
        "endpoints": {
            "llm": _safe_endpoint((endpoints or {}).get("llm") or os.environ.get("LLM_BASE_URL")),
            "parakeet": _safe_endpoint((endpoints or {}).get("parakeet") or os.environ.get("PARAKEET_URL")),
            "nemotron": _safe_endpoint((endpoints or {}).get("nemotron") or os.environ.get("NEMOTRON_URL")),
        },
        "prompts": prompt_versions,
        "schema_versions": schema_versions,
        "implementation_sha256": {
            key: _sha256_file(path) for key, path in scripts.items() if path.is_file()
        },
        "configuration": recorded_config,
        "stages": [],
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_artifact_schema_fingerprints(manifest: dict[str, Any], output: Path) -> None:
    fingerprints = manifest.setdefault("schema_fingerprints", {})
    configs = {
        "fact_extraction": output / "facts" / "run_config.json",
        "speaker_names": output / "speaker_names" / "run_config.json",
    }
    for name, path in configs.items():
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
            schema = config.get("schema")
            if schema is None:
                continue
            canonical = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            fingerprints[name] = {
                "version": config.get("version") or config.get("schema_version"),
                "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            }
        except (OSError, ValueError, TypeError):
            continue
