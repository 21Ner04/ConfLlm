import copy
import json
import re
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_report(folder, review):
    export_id = uuid4().hex
    target = folder / "review_exports" / export_id
    target.mkdir(parents=True, exist_ok=False)

    # Снимок конкретной сохранённой редакции.
    write_json(target / "review_snapshot.json", review)

    names = {}
    participants = []
    for speaker_id, entry in review["speakers"].items():
        name = entry.get("name")
        status = entry.get("status")
        if name and status in {"confirmed", "inferred"}:
            display = (
                name if status == "confirmed"
                else name + " (предположительно)"
            )
            names[speaker_id.casefold()] = display
            participants.append(f"{speaker_id}: {display}")
        else:
            participants.append(f"{speaker_id}: имя не установлено")

    def render(text):
        return re.sub(
            r"\bspeaker_\d+\b",
            lambda match: names.get(
                match.group(0).casefold(), match.group(0)
            ),
            text,
            flags=re.IGNORECASE,
        )

    items = copy.deepcopy(review["items"])
    for item in items:
        item["text"] = render(item["text"])
        if item.get("owner"):
            item["owner"] = render(item["owner"])
        # evidence не меняем: исходные цитаты и speaker_id сохраняются.

    limitations = list(review.get("limitations", []))
    limitations.extend([
        f"Экспорт сохранённой редакции {review['revision']}",
        f"Время сохранения редакции: {review.get('saved_at')}",
        "Подтверждение имени пользователем не означает утверждения всего протокола",
        "Свободный текст пунктов не переписывается при смене имени участника",
    ])
    if participants:
        limitations.append("Участники: " + "; ".join(participants))

    facts = {
        "meeting_id": review["meeting_id"],
        "status": (
            "partial_needs_review" if review.get("failed_blocks")
            else "draft_needs_review"
        ),
        "items": items,
        "warnings": review.get("warnings", []),
        "failed_blocks": review.get("failed_blocks", []),
        "limitations": limitations,
    }
    source = target / "export_input.json"
    write_json(source, facts)

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "export_meeting_report.py"),
                "--source", str(source),
                "--out", str(target / "reports"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=False,
        )
        (target / "export.log").write_text(
            result.stdout + "\n" + result.stderr,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError("Экспорт завершился с ошибкой")

        for extension in ["docx", "pdf"]:
            path = target / "reports" / f"protocol.{extension}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError("Экспорт не создал итоговые документы")

        write_json(target / "manifest.json", {
            "export_id": export_id,
            "revision": review["revision"],
            "status": "completed",
        })
    except Exception:
        write_json(target / "export_status.json", {"status": "failed"})
        raise

    return export_id
