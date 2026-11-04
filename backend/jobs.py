import re
import subprocess
import sys
import threading
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from .job_store import JOBS, ROOT, acquire, pipeline_state, reconcile, save

MAX_BYTES = 1024 * 1024 * 1024
router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("")
def upload_video(file: UploadFile = File(...)):
    from pathlib import Path

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}:
        file.file.close()
        raise HTTPException(422, "Поддерживаются MP4, MKV, MOV, WEBM, AVI, M4V")
    if file.size is not None and file.size > MAX_BYTES:
        file.file.close()
        raise HTTPException(413, "Лимит этой версии: 1 ГиБ")

    global_lock = acquire(JOBS / "execution.lock")
    if global_lock is None:
        file.file.close()
        raise HTTPException(409, "Уже выполняется обработка другой записи")

    job_id = uuid4().hex
    folder = JOBS / job_id
    job_lock = None
    process = None
    data = {
        "job_id": job_id,
        "meeting_id": "web_" + job_id,
        "filename": file.filename,
        "status": "uploading",
        "worker_version": 2,
    }

    try:
        folder.mkdir()
        job_lock = acquire(folder / "execution.lock")
        if job_lock is None:
            raise RuntimeError("Не удалось получить блокировку задания")

        video = folder / ("source" + suffix)
        data["video_path"] = str(video)
        save(folder / "job.json", data)

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
        data["status"] = "queued"
        save(folder / "job.json", data)

        with (folder / "pipeline.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable, "-u", "-m", "backend.job_worker",
                    job_id,
                    str(global_lock.fileno()),
                    str(job_lock.fileno()),
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(global_lock.fileno(), job_lock.fileno()),
            )

        # Только сбор кода завершения; состояние пишет отдельный worker.
        threading.Thread(target=process.wait, daemon=True).start()
        return {"job_id": job_id, "meeting_id": data["meeting_id"]}

    except Exception:
        # Если worker уже создан, не удаляем его входной файл.
        if process is None and folder.is_dir():
            data["status"] = "failed"
            data["error_type"] = "UploadOrLaunchFailed"
            save(folder / "job.json", data)
            if data.get("video_path"):
                Path(data["video_path"]).unlink(missing_ok=True)
        raise
    finally:
        file.file.close()
        if job_lock is not None:
            job_lock.close()
        global_lock.close()


@router.get("/{job_id}")
def job_status(job_id: str):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(400, "Недопустимый job_id")
    folder = JOBS / job_id
    if not (folder / "job.json").is_file():
        raise HTTPException(404, "Задание не найдено")

    data = reconcile(folder)
    pipeline = pipeline_state(data["meeting_id"])
    if pipeline:
        data["run_id"] = pipeline["run_id"]
        data["pipeline_status"] = pipeline["status"]
        data["current_step"] = pipeline.get("current_step")
        data["completed_steps"] = len(pipeline.get("completed_steps", []))
        if pipeline.get("error_type"):
            data["error_type"] = pipeline["error_type"]

    if data["status"] == "failed":
        log = folder / "pipeline.log"
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 5000))
                data["log_tail"] = stream.read().decode("utf-8", errors="replace")
    return data
