import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.quality_report import collect_quality_report
from backend.run_manifest import (
    build_run_manifest,
    record_artifact_schema_fingerprints,
    utc_now,
)

ROOT = Path(__file__).resolve().parent


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Видео → Parakeet + Nemotron → транскрипт → черновик фактов."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--meeting-id", required=True)
    parser.add_argument("--parakeet-url")
    parser.add_argument("--nemotron-url")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.meeting_id):
        parser.error("--meeting-id: 1–80 латинских букв, цифр, _ или -")

    video = args.video.expanduser().resolve()
    if not video.is_file():
        parser.error(f"Видео не найдено: {video}")

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)

    parakeet_url = (
        args.parakeet_url or os.environ.get("PARAKEET_URL", "")
    ).strip().rstrip("/")
    nemotron_url = (
        args.nemotron_url or os.environ.get("NEMOTRON_URL", "")
    ).strip().rstrip("/")

    for label, url in [
        ("Parakeet", parakeet_url),
        ("Nemotron", nemotron_url),
    ]:
        if not url.startswith(("http://", "https://")):
            parser.error(f"Укажите HTTP(S)-адрес сервера {label}")

    for variable in ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"]:
        if not os.environ.get(variable):
            parser.error(f"Не задана переменная {variable} в окружении или .env")

    for program in ["ffmpeg", "curl"]:
        if shutil.which(program) is None:
            parser.error(f"Не найдена программа {program}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "outputs" / args.meeting_id / run_id
    work = ROOT / "data/meetings" / args.meeting_id / "runs" / run_id
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)

    audio = work / "audio.wav"
    vad = output / "vad.json"
    chunks = work / "asr_chunks"
    asr = output / "parakeet"
    diar = output / "nemotron_response.json"
    aligned = output / "transcript"
    facts = output / "facts"
    speaker_names = output / "speaker_names"
    status_path = output / "pipeline_status.json"
    manifest_path = output / "run_manifest.json"
    started_at = utc_now()
    run_config = {
        "audio": {"sample_rate_hz": 16000, "channels": 1, "format": "pcm_s16le"},
        "vad": {"implementation": "Silero VAD", "device": "cpu"},
        "asr_chunk_max_seconds": 20,
        "llm_temperature": 0,
        "llm_thinking_enabled": False,
    }
    run_manifest = build_run_manifest(
        root=ROOT,
        run_id=run_id,
        meeting_id=args.meeting_id,
        video_path=video,
        started_at=started_at,
        endpoints={
            "llm": os.environ.get("LLM_BASE_URL"),
            "parakeet": parakeet_url,
            "nemotron": nemotron_url,
        },
        config=run_config,
    )

    status = {
        "meeting_id": args.meeting_id,
        "run_id": run_id,
        "video": str(video),
        "status": "running",
        "started_at": started_at,
        "completed_steps": [],
        "current_step": None,
        "stage_metrics": [],
    }
    save_json(manifest_path, run_manifest)
    save_json(status_path, status)

    def persist_status():
        run_manifest["stages"] = status["stage_metrics"]
        run_manifest["status"] = status["status"]
        run_manifest["finished_at"] = status.get("finished_at")
        record_artifact_schema_fingerprints(run_manifest, output)
        save_json(manifest_path, run_manifest)
        save_json(status_path, status)

    def run(step, command):
        status["current_step"] = step
        stage = {
            "name": step,
            "status": "running",
            "started_at": utc_now(),
        }
        status["stage_metrics"].append(stage)
        persist_status()
        print(f"\n=== {step} ===", flush=True)
        started = time.perf_counter()
        try:
            result = subprocess.run(command, cwd=ROOT, check=True)
            stage["return_code"] = result.returncode
        except subprocess.CalledProcessError as error:
            stage["status"] = "failed"
            stage["return_code"] = error.returncode
            stage["error_type"] = type(error).__name__
            raise
        except Exception as error:
            stage["status"] = "failed"
            stage["error_type"] = type(error).__name__
            raise
        finally:
            stage["duration_seconds"] = round(time.perf_counter() - started, 3)
            stage["finished_at"] = utc_now()
            persist_status()

    def complete(step):
        status["completed_steps"].append(step)
        if status["stage_metrics"] and status["stage_metrics"][-1]["name"] == step:
            if status["stage_metrics"][-1]["status"] == "failed":
                status["stage_metrics"][-1]["status"] = "completed_with_fallback"
            else:
                status["stage_metrics"][-1]["status"] = "completed"
            status["stage_metrics"][-1].setdefault("finished_at", utc_now())
        persist_status()

    python = sys.executable

    try:
        step = "1/10 Извлечение аудио"
        run(step, [
            "ffmpeg", "-nostdin", "-n", "-i", str(video),
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(audio),
        ])
        complete(step)

        step = "2/10 Silero VAD"
        run(step, [
            python, "detect_speech_silero.py",
            "--source", str(audio), "--out", str(vad),
        ])
        complete(step)

        step = "3/10 Подготовка фрагментов"
        run(step, [
            python, "prepare_asr_chunks.py",
            "--vad", str(vad), "--out", str(chunks),
        ])
        complete(step)

        step = "4/10 Parakeet"
        run(step, [
            python, "transcribe_parakeet_chunks.py",
            "--manifest", str(chunks / "manifest.json"),
            "--out", str(asr), "--url", parakeet_url,
        ])
        complete(step)

        step = "5/10 Диаризация Nemotron"
        partial = output / "nemotron_response.partial.json"
        run(step, [
            "curl", "--silent", "--show-error", "--fail-with-body",
            "--connect-timeout", "10", "--max-time", "1800",
            "-F", f"file=@{audio}",
            f"{nemotron_url}/v1/audio/diarizations",
            "-o", str(partial),
        ])

        response = json.loads(partial.read_text(encoding="utf-8"))
        if not isinstance(response.get("segments"), list) or not response["segments"]:
            raise ValueError("Nemotron не вернул непустой список segments")
        partial.replace(diar)
        complete(step)

        step = "6/10 Объединение текста и говорящих"
        run(step, [
            python, "merge_asr_transcript.py", "--backend", "parakeet",
            "--meeting-id", args.meeting_id,
            "--asr", str(asr / "transcript_original_time.json"),
            "--diarization", str(diar), "--vad", str(vad),
            "--out", str(aligned),
        ])
        complete(step)

        step = "7/10 Определение предполагаемых имён"
        try:
            run(step, [
                python, "resolve_speaker_names.py",
                "--source", str(aligned / "aligned_transcript.json"),
                "--out", str(speaker_names),
            ])
        except subprocess.CalledProcessError:
            from speaker_names_fallback import write_fallback

            warning = (
                "Автоматическое определение имён завершилось ошибкой. "
                "Используются исходные идентификаторы спикеров."
            )
            write_fallback(
                aligned / "aligned_transcript.json",
                speaker_names,
                warning,
            )
            status.setdefault("warnings", []).append(warning)
            print("ПРЕДУПРЕЖДЕНИЕ:", warning, flush=True)
        complete(step)

        step = "8/10 Извлечение фактов через LLM"
        run(step, [
            python, "extract_meeting_facts.py",
            "--source", str(aligned / "aligned_transcript.json"),
            "--out", str(facts),
        ])

        if status.get("warnings"):
            facts_path = facts / "facts.json"
            facts_data = json.loads(facts_path.read_text(encoding="utf-8"))
            for warning in status["warnings"]:
                facts_data.setdefault("warnings", []).append({
                    "stage": "speaker_names",
                    "warning": warning,
                })
            save_json(facts_path, facts_data)

            markdown_path = facts / "facts.md"
            with markdown_path.open("a", encoding="utf-8") as stream:
                stream.write("\n\n## Предупреждения определения имён\n\n")
                for warning in status["warnings"]:
                    stream.write("- " + warning + "\n")

            complete(step)

        step = "9/10 Экспорт DOCX и PDF"
        run(step, [
            python, "export_meeting_report.py",
            "--source", str(facts / "facts.json"),
            "--speaker-names", str(speaker_names / "speaker_names.json"),
        ])
        complete(step)

        step = "10/10 Сохранение итогового статуса"
        status["current_step"] = step
        final_stage_started = time.perf_counter()
        final_stage = {
            "name": step,
            "status": "running",
            "started_at": utc_now(),
        }
        status["stage_metrics"].append(final_stage)
        persist_status()
        for name in [
            "facts.json", "facts.md",
            "reports/protocol.docx", "reports/protocol.pdf",
        ]:
            if not (facts / name).is_file():
                raise ValueError(f"Не создан итоговый файл {name}")

        if not (speaker_names / "speaker_names.json").is_file():
            raise ValueError("Не создана таблица имён участников")

        status["artifacts"] = {
            "speaker_names_json": str(speaker_names / "speaker_names.json"),
            "transcript_json": str(aligned / "aligned_transcript.json"),
            "transcript_txt": str(aligned / "aligned_transcript.txt"),
            "facts_json": str(facts / "facts.json"),
            "facts_markdown": str(facts / "facts.md"),
            "protocol_docx": str(facts / "reports/protocol.docx"),
            "protocol_pdf": str(facts / "reports/protocol.pdf"),
        }
        status["quality_report"] = str(output / "quality_report.json")
        quality_report = collect_quality_report(
            output, work, status["stage_metrics"]
        )
        final_stage["status"] = "completed"
        final_stage["duration_seconds"] = round(time.perf_counter() - final_stage_started, 3)
        final_stage["finished_at"] = utc_now()
        quality_report["stages"] = status["stage_metrics"]
        save_json(output / "quality_report.json", quality_report)
        status["completed_steps"].append(step)
        status["status"] = "completed_needs_review"
        status["current_step"] = None
        status["finished_at"] = utc_now()
        if status["stage_metrics"]:
            status["stage_metrics"][-1]["status"] = "completed"
            status["stage_metrics"][-1]["finished_at"] = status["finished_at"]
        persist_status()

        print("\nГотово: получен черновик, требующий проверки.", flush=True)
        print("Транскрипт:", aligned)
        print("Факты:", facts)

    except (Exception, KeyboardInterrupt) as error:
        status["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        status["error_type"] = type(error).__name__
        status["finished_at"] = utc_now()
        if status["stage_metrics"] and status["stage_metrics"][-1]["status"] == "running":
            status["stage_metrics"][-1]["status"] = "failed"
            status["stage_metrics"][-1]["error_type"] = type(error).__name__
            status["stage_metrics"][-1]["finished_at"] = status["finished_at"]
        try:
            partial_report = collect_quality_report(
                output, work, status["stage_metrics"]
            )
            partial_report["pipeline_error"] = {
                "stage": status.get("current_step"),
                "error_type": type(error).__name__,
            }
            status["quality_report"] = str(output / "quality_report.json")
            save_json(output / "quality_report.json", partial_report)
        except Exception:
            pass
        persist_status()
        print(f"\nОбработка остановлена на этапе: {status['current_step']}",
              file=sys.stderr)
        print(f"Статус и промежуточные результаты: {output}", file=sys.stderr)
        if isinstance(error, KeyboardInterrupt):
            raise SystemExit(130)
        raise


if __name__ == "__main__":
    main()
