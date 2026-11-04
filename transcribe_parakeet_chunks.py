import argparse
import json
import math
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Распознавание VAD-фрагментов через HTTP-сервер Parakeet."
)
parser.add_argument(
    "--manifest", type=Path,
    default=Path("data/meetings/meeting_001/asr_chunks_v1/manifest.json"),
)
parser.add_argument(
    "--out", type=Path, default=Path("outputs/meeting_001_parakeet_vad_v1")
)
parser.add_argument("--url", default=os.environ.get("PARAKEET_URL", ""))
args = parser.parse_args()

MANIFEST = args.manifest
OUT = args.out
URL = args.url.strip().rstrip("/")
if not URL:
    parser.error("Укажите --url или задайте PARAKEET_URL")
MODEL = "parakeet-tdt-0.6b-v3.oss-align.q8_0"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


manifest = read_json(MANIFEST)
OUT.mkdir(parents=True, exist_ok=True)

run_config = {
    "manifest": manifest,
    "model": MODEL,
    "response_format": "verbose_json",
}
config_path = OUT / "run_config.json"

if config_path.exists():
    if read_json(config_path) != run_config:
        raise SystemExit("Настройки отличаются от сохранённых. Нужна новая папка.")
else:
    save_json(config_path, run_config)

all_words = []
chunk_records = []

for position, chunk in enumerate(manifest["chunks"], start=1):
    chunk_id = chunk["id"]
    audio = Path(chunk["audio_path"])
    response_path = OUT / f"chunk_{chunk_id:04d}.json"
    timing_path = OUT / f"chunk_{chunk_id:04d}_seconds.txt"

    if not audio.is_file():
        raise SystemExit(f"Не найден WAV: {audio}")

    print(
        f"[{position}/{len(manifest['chunks'])}] chunk_{chunk_id:04d}",
        flush=True,
    )

    if response_path.exists():
        response = read_json(response_path)
        print("  Используем сохранённый ответ", flush=True)
    else:
        partial = OUT / f"chunk_{chunk_id:04d}.partial.json"
        command = [
            "curl", "--silent", "--show-error", "--fail-with-body",
            "--connect-timeout", "10", "--max-time", "300",
            f"{URL}/v1/audio/transcriptions",
            "-F", f"file=@{audio}",
            "-F", f"model={MODEL}",
            "-F", "response_format=verbose_json",
            "-o", str(partial),
            "-w", "%{time_total}",
        ]

        completed = subprocess.run(command, capture_output=True, text=True)

        if completed.returncode:
            print(completed.stderr)
            if partial.exists():
                print(partial.read_text(encoding="utf-8", errors="replace"))
            raise SystemExit(
                f"Ошибка на chunk_{chunk_id:04d}. "
                "Исправь причину и повтори запуск."
            )

        response = read_json(partial)
        if not isinstance(response.get("words"), list):
            raise SystemExit(f"Нет массива words: {partial}")

        partial.replace(response_path)
        timing_path.write_text(
            completed.stdout.strip() + "\n", encoding="utf-8"
        )

    words = response.get("words")
    if not isinstance(words, list):
        raise SystemExit(f"Некорректный ответ: {response_path}")

    if response.get("text", "").strip() and not words:
        raise SystemExit(f"Есть текст, но нет таймкодов слов: {response_path}")

    offset = chunk["source_start_seconds"]
    duration = chunk["duration_seconds"]

    for word_index, word in enumerate(words):
        local_start = float(word["start"])
        local_end = float(word["end"])

        if not (
            math.isfinite(local_start)
            and math.isfinite(local_end)
            and 0 <= local_start <= local_end <= duration + 0.1
        ):
            raise SystemExit(
                f"Некорректные границы в {response_path}: {word}"
            )

        flags = list(chunk["review_flags"])
        if local_end > duration:
            flags.append("timestamp_exceeds_chunk_end")

        all_words.append({
            "word": word["word"],
            "start": offset + local_start,
            "end": offset + local_end,
            "local_start": local_start,
            "local_end": local_end,
            "chunk_id": chunk_id,
            "word_index_in_chunk": word_index,
            "review_flags": flags,
        })

    elapsed = (
        float(timing_path.read_text().strip())
        if timing_path.exists() else None
    )
    chunk_records.append({
        "chunk_id": chunk_id,
        "response_path": response_path.as_posix(),
        "source_start_seconds": offset,
        "source_end_seconds": chunk["source_end_seconds"],
        "text": response.get("text", ""),
        "request_seconds": elapsed,
        "review_flags": chunk["review_flags"],
    })

    print(f"  Слов: {len(words)}; HTTP: {elapsed} с", flush=True)

all_words.sort(
    key=lambda word: (
        word["start"], word["chunk_id"], word["word_index_in_chunk"]
    )
)

result = {
    "source_audio": manifest["source_audio"],
    "model": MODEL,
    "manifest_path": MANIFEST.as_posix(),
    "time_reference": "original_audio_seconds",
    "manually_verified": False,
    "chunks": chunk_records,
    "words": all_words,
    "statistics": {
        "chunks": len(chunk_records),
        "words": len(all_words),
        "http_request_seconds_sum": sum(
            item["request_seconds"] or 0 for item in chunk_records
        ),
    },
}

target = OUT / "transcript_original_time.json"
save_json(target, result)

print(f"\nСохранено: {target}")
print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))
