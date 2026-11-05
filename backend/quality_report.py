"""Collect non-accuracy health indicators from pipeline artifacts."""

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def collect_quality_report(output: Path, work: Path, stages: list[dict]) -> dict[str, Any]:
    vad = _read_json(output / "vad.json")
    manifest = _read_json(work / "asr_chunks" / "manifest.json")
    asr = _read_json(output / "parakeet" / "transcript_original_time.json")
    diarization = _read_json(output / "nemotron_response.json")
    transcript = _read_json(output / "transcript" / "aligned_transcript.json")
    names = _read_json(output / "speaker_names" / "speaker_names.json")
    names_status = _read_json(output / "speaker_names" / "status.json")
    facts = _read_json(output / "facts" / "facts.json")
    extraction_status = _read_json(output / "facts" / "extraction_status.json")

    utterances = transcript.get("utterances", [])
    items = facts.get("items", [])
    speakers = names.get("speakers", {})
    kinds = Counter(item.get("kind", "unknown") for item in items if isinstance(item, dict))
    statuses = Counter(entry.get("status", "unknown") for entry in speakers.values() if isinstance(entry, dict))
    flags = Counter(
        flag
        for utterance in utterances if isinstance(utterance, dict)
        for flag in utterance.get("review_flags", [])
    )
    chunks = manifest.get("chunks", [])

    return {
        "report_schema_version": "pipeline_quality_v1",
        "interpretation": "Coverage and operational diagnostics; not an accuracy score.",
        "stages": stages,
        "audio": {
            "duration_seconds": vad.get("duration_seconds"),
            "speech_seconds_detected": vad.get("speech_seconds"),
            "vad_interval_count": len(vad.get("segments", [])),
        },
        "chunking": {
            "chunk_count": len(chunks),
            "total_chunk_seconds": round(sum(float(item.get("duration_seconds", 0)) for item in chunks), 3),
            "forced_split_count": sum(
                "forced_split_inside_vad_interval" in item.get("review_flags", [])
                for item in chunks if isinstance(item, dict)
            ),
        },
        "asr": {
            "model": asr.get("model"),
            "word_count": len(asr.get("words", [])),
        },
        "diarization": {
            "segment_count": len(diarization.get("segments", [])),
        },
        "transcript": {
            "utterance_count": len(utterances),
            "unknown_speaker_count": sum(item.get("speaker") == "unknown" for item in utterances if isinstance(item, dict)),
            "review_flag_counts": dict(flags),
        },
        "speaker_names": {
            "status_counts": dict(statuses),
            "total_windows": names_status.get("total_windows"),
            "failed_windows": len(names_status.get("failed_windows", [])),
            "attempt_count": sum(names_status.get("attempts_by_window", {}).values()),
            "retry_count": sum(max(0, attempts - 1) for attempts in names_status.get("attempts_by_window", {}).values()),
        },
        "fact_extraction": {
            "item_count": len(items),
            "item_kind_counts": dict(kinds),
            "failed_block_count": len(facts.get("failed_blocks", [])),
            "warning_count": len(facts.get("warnings", [])),
            "total_blocks": facts.get("total_blocks"),
            "attempt_count": sum(extraction_status.get("attempts_by_block", {}).values()),
            "retry_count": sum(max(0, attempts - 1) for attempts in extraction_status.get("attempts_by_block", {}).values()),
        },
        "exports": {
            "docx_exists": (output / "facts" / "reports" / "protocol.docx").is_file(),
            "pdf_exists": (output / "facts" / "reports" / "protocol.pdf").is_file(),
        },
    }
