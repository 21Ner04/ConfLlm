import os
import subprocess
import sys

from .job_store import JOBS, ROOT, read, save


def main():
    job_id, global_fd, job_fd = sys.argv[1:]
    descriptors = (int(global_fd), int(job_fd))
    folder = JOBS / job_id
    path = folder / "job.json"
    data = read(path)

    try:
        data["status"] = "running"
        data["worker_pid"] = os.getpid()
        save(path, data)

        process = subprocess.Popen(
            [
                sys.executable, "-u", str(ROOT / "run_pipeline.py"),
                "--video", data["video_path"],
                "--meeting-id", data["meeting_id"],
            ],
            cwd=ROOT,
            # Пайплайн тоже удерживает блокировки.
            pass_fds=descriptors,
        )
        data["pipeline_pid"] = process.pid
        save(path, data)

        code = process.wait()
        data["returncode"] = code
        data["status"] = "completed" if code == 0 else "failed"
        if code != 0:
            data["error_type"] = "PipelineFailed"
        save(path, data)
    except Exception as error:
        data["status"] = "failed"
        data["error_type"] = type(error).__name__
        save(path, data)
        raise
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


if __name__ == "__main__":
    main()
