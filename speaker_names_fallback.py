import json
from pathlib import Path


def write_fallback(source, out, warning):
    source = Path(source).resolve()
    out = Path(out)
    transcript = json.loads(source.read_text(encoding="utf-8"))
    speakers = sorted({
        item["speaker"] for item in transcript["utterances"]
    } - {"unknown"})

    out.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_version": "speaker_names_v1",
        "status": "needs_review",
        "resolution_status": "failed_fallback",
        "source_transcript": str(source),
        "manually_verified": False,
        "warnings": [warning],
        "limitations": [
            "Автоматические имена не получены",
            "Можно указать и подтвердить имена вручную в редакторе",
        ],
        "speakers": {
            speaker: {
                "speaker_id": speaker,
                "name": None,
                "display_name": speaker,
                "status": "unresolved",
                "candidates": [],
            }
            for speaker in speakers
        },
    }
    path = out / "speaker_names.json"
    temporary = out / "speaker_names.fallback.tmp"
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return result
