import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
app = FastAPI(title="ConfLlm Review API", version="0.1.0")


class Speaker(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str | None = Field(default=None, max_length=100)
    status: Literal["inferred", "confirmed", "unresolved", "conflict"]


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: int
    kind: Literal[
        "discussion", "proposal", "decision",
        "task", "open_question", "risk"
    ]
    text: str = Field(min_length=1, max_length=20000)
    owner: str | None
    deadline: str | None


class ReviewUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision: int = Field(ge=0)
    speakers: dict[str, Speaker]
    items: list[Item]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_folder(meeting_id, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", meeting_id):
        raise HTTPException(400, "Недопустимый meeting_id")
    if not re.fullmatch(r"\d{8}T\d{6}_\d{6}Z", run_id):
        raise HTTPException(400, "Недопустимый run_id")

    folder = OUTPUTS / meeting_id / run_id
    if not folder.is_dir():
        raise HTTPException(404, "Запуск не найден")
    if not folder.resolve().is_relative_to(OUTPUTS.resolve()):
        raise HTTPException(400, "Недопустимый путь")
    return folder


def initial_review(folder, meeting_id, run_id):
    facts_path = folder / "facts/facts.json"
    transcript_path = folder / "transcript/aligned_transcript.json"
    if not facts_path.is_file() or not transcript_path.is_file():
        raise HTTPException(409, "Результат обработки ещё не готов")

    facts = read_json(facts_path)
    transcript = read_json(transcript_path)
    names_path = folder / "speaker_names/speaker_names.json"
    names = read_json(names_path).get("speakers", {}) if names_path.is_file() else {}

    speakers = {}
    for speaker_id in sorted({u["speaker"] for u in transcript["utterances"]}):
        entry = names.get(speaker_id, {})
        name = entry.get("name")
        status = entry.get("status", "unresolved")
        speakers[speaker_id] = {
            "name": name,
            "status": status,
        }

    items = []
    for item in facts["items"]:
        items.append({
            "id": item["id"],
            "kind": item["kind"],
            "text": item["text"],
            "owner": item.get("owner"),
            "deadline": item.get("deadline"),
            "original_text": item["text"],
            "evidence": item.get("evidence", []),
            "edited": False,
        })

    return {
        "schema_version": "reviewed_protocol_v1",
        "meeting_id": meeting_id,
        "run_id": run_id,
        "revision": 0,
        "status": "draft",
        "source_status": facts.get("status"),
        "source_facts_json": str(facts_path),
        "source_transcript_json": str(transcript_path),
        "source_speaker_names_json": str(names_path) if names_path.is_file() else None,
        "warnings": facts.get("warnings", []),
        "failed_blocks": facts.get("failed_blocks", []),
        "limitations": facts.get("limitations", []),
        "saved_at": None,
        "speakers": speakers,
        "items": items,
    }


def load_review(folder, meeting_id, run_id):
    path = folder / "reviewed_protocol.json"
    if path.is_file():
        return read_json(path)
    return initial_review(folder, meeting_id, run_id)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meetings")
def meetings():
    results = []
    if OUTPUTS.is_dir():
        for status_path in sorted(
            OUTPUTS.glob("*/*/pipeline_status.json"), reverse=True
        ):
            if not status_path.resolve().is_relative_to(OUTPUTS.resolve()):
                continue
            try:
                data = read_json(status_path)
                results.append({
                    "meeting_id": data["meeting_id"],
                    "run_id": data["run_id"],
                    "status": data["status"],
                    "current_step": data.get("current_step"),
                })
            except (OSError, ValueError, KeyError):
                continue
    return results


@app.get("/api/meetings/{meeting_id}/{run_id}/review")
def get_review(meeting_id: str, run_id: str):
    folder = run_folder(meeting_id, run_id)
    return load_review(folder, meeting_id, run_id)


@app.get("/api/meetings/{meeting_id}/{run_id}/transcript")
def get_transcript(meeting_id: str, run_id: str):
    folder = run_folder(meeting_id, run_id)
    path = folder / "transcript/aligned_transcript.json"
    if not path.is_file():
        raise HTTPException(409, "Транскрипт ещё не готов")
    return read_json(path)


@app.put("/api/meetings/{meeting_id}/{run_id}/review")
def put_review(meeting_id: str, run_id: str, update: ReviewUpdate):
    folder = run_folder(meeting_id, run_id)

    # Один процесс API; блокируем параллельные сохранения этой редакции.
    lock_path = folder / ".review.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        raise HTTPException(409, "Редакция сохраняется. Повтори запрос позже")

    try:
        with lock:
            current = load_review(folder, meeting_id, run_id)
            if update.revision != current["revision"]:
                raise HTTPException(409, "Редакция изменилась. Загрузи её заново")

            if set(update.speakers) != set(current["speakers"]):
                raise HTTPException(422, "Нельзя менять набор speaker_id")

            for speaker in update.speakers.values():
                if speaker.name is not None:
                    speaker.name = speaker.name.strip()
                    if not speaker.name:
                        raise HTTPException(422, "Имя не должно быть пустым")
                if speaker.status in {"confirmed", "inferred"} and not speaker.name:
                    raise HTTPException(422, "Для этого статуса необходимо имя")
                if speaker.status in {"unresolved", "conflict"} and speaker.name:
                    raise HTTPException(422, "Сначала выбери статус имени")

            original = {item["id"]: item for item in current["items"]}
            received = [item.id for item in update.items]
            if len(received) != len(set(received)) or set(received) != set(original):
                raise HTTPException(422, "Нельзя добавлять или удалять ID пунктов")

            for edited in update.items:
                if not edited.text.strip():
                    raise HTTPException(422, "Пустая формулировка")
                if edited.owner is not None:
                    if edited.owner not in current["speakers"] or edited.owner == "unknown":
                        raise HTTPException(422, "Неизвестный исполнитель")
                if edited.kind != "task" and (
                    edited.owner is not None or edited.deadline is not None
                ):
                    raise HTTPException(422, "Исполнитель и срок допустимы только для task")
                if edited.deadline is not None and not edited.deadline.strip():
                    raise HTTPException(422, "Пустой срок")

                target = original[edited.id]
                values = edited.model_dump()
                changed = any(target.get(k) != v for k, v in values.items())
                target.update(values)
                if changed:
                    target["edited"] = True
                    target["edited_at"] = datetime.now(timezone.utc).isoformat()

            current["speakers"] = {
                key: value.model_dump() for key, value in update.speakers.items()
            }
            current["revision"] += 1
            current["saved_at"] = datetime.now(timezone.utc).isoformat()
            path = folder / "reviewed_protocol.json"
            temporary = folder / "reviewed_protocol.json.tmp"
            temporary.write_text(
                json.dumps(current, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
            return current
    finally:
        lock_path.unlink(missing_ok=True)

@app.get("/", include_in_schema=False)
def editor_page():
    from fastapi.responses import FileResponse
    return FileResponse(
        Path(__file__).with_name("editor.html"),
        media_type="text/html",
    )


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision: int = Field(ge=1)


@app.post("/api/meetings/{meeting_id}/{run_id}/export")
def export_review(meeting_id: str, run_id: str, request: ExportRequest):
    from .report_export import build_report

    folder = run_folder(meeting_id, run_id)
    saved = folder / "reviewed_protocol.json"
    if not saved.is_file():
        raise HTTPException(409, "Сначала сохрани редакцию")

    review = read_json(saved)
    if review["revision"] != request.revision:
        raise HTTPException(409, "Редакция изменилась. Открой её заново")

    lock_path = folder / ".export.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        raise HTTPException(409, "Экспорт уже выполняется")

    try:
        with lock:
            export_id = build_report(folder, review)
    except Exception:
        raise HTTPException(
            500,
            "Экспорт не выполнен. Проверь журнал в папке review_exports",
        )
    finally:
        lock_path.unlink(missing_ok=True)

    base = f"/api/meetings/{meeting_id}/{run_id}/exports/{export_id}"
    return {
        "revision": review["revision"],
        "export_id": export_id,
        "docx": base + "/docx",
        "pdf": base + "/pdf",
    }


@app.get("/api/meetings/{meeting_id}/{run_id}/exports/{export_id}/{format}")
def download_report(
    meeting_id: str, run_id: str, export_id: str, format: str
):
    from fastapi.responses import FileResponse

    folder = run_folder(meeting_id, run_id)
    if not re.fullmatch(r"[a-f0-9]{32}", export_id):
        raise HTTPException(400, "Недопустимый идентификатор экспорта")
    if format not in {"docx", "pdf"}:
        raise HTTPException(400, "Формат: docx или pdf")

    target = folder / "review_exports" / export_id
    if not target.resolve().is_relative_to(folder.resolve()):
        raise HTTPException(400, "Недопустимый путь")
    manifest = target / "manifest.json"
    if not manifest.is_file():
        raise HTTPException(404, "Готовый экспорт не найден")

    metadata = read_json(manifest)
    path = target / "reports" / f"protocol.{format}"
    if not path.is_file():
        raise HTTPException(404, "Документ не найден")

    media_type = (
        "application/pdf" if format == "pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(
        path,
        media_type=media_type,
        filename=f"{meeting_id}_revision_{metadata['revision']}.{format}",
    )
