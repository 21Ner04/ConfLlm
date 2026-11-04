import json
import re
import subprocess
import sys
import threading
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "web_jobs"
ACTIVE = JOBS / "active.lock"
MAX_BYTES = 1024 * 1024 * 1024
router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def release(job_id):
    try:
        if ACTIVE.read_text(encoding="utf-8") == job_id:
            ACTIVE.unlink()
    except FileNotFoundError:
        pass


def watch(process, log, folder, data):
    try:
        code = process.wait()
        data["status"] = "completed" if code == 0 else "failed"
        data["returncode"] = code
        save(folder / "job.json", data)
    finally:
        log.close()
        release(data["job_id"])


@router.post("")
def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}:
        raise HTTPException(422, "Поддерживаются MP4, MKV, MOV, WEBM, AVI, M4V")
    if file.size is not None and file.size > MAX_BYTES:
        raise HTTPException(413, "Лимит этой версии: 1 ГиБ")

    JOBS.mkdir(exist_ok=True)
    job_id = uuid4().hex
    try:
        with ACTIVE.open("x", encoding="utf-8") as lock:
            lock.write(job_id)
    except FileExistsError:
        raise HTTPException(
            409,
            "Уже есть активное задание. Дождись завершения. "
            "После аварийного перезапуска может потребоваться проверка active.lock.",
        )

    folder = JOBS / job_id
    folder.mkdir()
    video = folder / ("source" + suffix)
    data = {
        "job_id": job_id,
        "meeting_id": "web_" + job_id,
        "filename": file.filename,
        "status": "uploading",
    }
    log = None
    process = None
    try:
        total = 0
        with video.open("xb") as destination:
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(413, "Лимит этой версии: 1 ГиБ")
                destination.write(chunk)
        if total == 0:
            raise HTTPException(422, "Файл пустой")

        data["size_bytes"] = total
        data["status"] = "running"
        save(folder / "job.json", data)
        log = (folder / "pipeline.log").open("wb")
        process = subprocess.Popen(
            [
                sys.executable, "-u", str(ROOT / "run_pipeline.py"),
                "--video", str(video),
                "--meeting-id", data["meeting_id"],
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        data["pid"] = process.pid
        save(folder / "job.json", data)
        thread = threading.Thread(
            target=watch,
            args=(process, log, folder, dict(data)),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id, "meeting_id": data["meeting_id"]}
    except Exception:
        if process is not None:
            process.terminate()
            process.wait()
        if log is not None:
            log.close()
        video.unlink(missing_ok=True)
        data["status"] = "failed"
        save(folder / "job.json", data)
        release(job_id)
        raise
    finally:
        file.file.close()


@router.get("/{job_id}")
def job_status(job_id: str):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(400, "Недопустимый job_id")
    folder = JOBS / job_id
    path = folder / "job.json"
    if not path.is_file():
        raise HTTPException(404, "Задание не найдено")

    data = read(path)
    statuses = sorted(
        (ROOT / "outputs" / data["meeting_id"]).glob("*/pipeline_status.json")
    )
    if statuses:
        pipeline = read(statuses[-1])
        data["run_id"] = pipeline["run_id"]
        data["pipeline_status"] = pipeline["status"]
        data["current_step"] = pipeline.get("current_step")
        data["completed_steps"] = len(pipeline.get("completed_steps", []))
        data["error_type"] = pipeline.get("error_type")

    if data["status"] == "failed":
        log = folder / "pipeline.log"
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 5000))
                data["log_tail"] = stream.read().decode("utf-8", errors="replace")
    return data
