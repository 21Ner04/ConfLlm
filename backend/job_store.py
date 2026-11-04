import fcntl
import json
import os
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "web_jobs"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def acquire(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    except Exception:
        stream.close()
        raise
    return stream


def pipeline_state(meeting_id):
    paths = sorted(
        (ROOT / "outputs" / meeting_id).glob("*/pipeline_status.json")
    )
    return read(paths[-1]) if paths else None


def reconcile(folder):
    path = folder / "job.json"
    data = read(path)
    if data["status"] in {"completed", "failed", "interrupted"}:
        return data

    lock = acquire(folder / "execution.lock")
    if lock is None:
        return data

    try:
        # После получения блокировки перечитываем актуальное состояние.
        data = read(path)
        if data["status"] in {"completed", "failed", "interrupted"}:
            return data
        pipeline = pipeline_state(data["meeting_id"])
        if pipeline and pipeline.get("status") == "completed_needs_review":
            data["status"] = "completed"
            data["recovered"] = True
        else:
            data["status"] = "failed"
            data["error_type"] = "WorkerInterrupted"
            data["error_message"] = (
                "Обработка больше не удерживает блокировку. "
                "Успешное завершение пайплайна не подтверждено."
            )
        save(path, data)
        return data
    finally:
        lock.close()
