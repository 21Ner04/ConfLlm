import argparse
import inspect
import json
import math
import time
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Распознавание VAD-фрагментов через GigaAM на CPU."
)
parser.add_argument(
    "--manifest", type=Path,
    default=Path("data/meetings/meeting_001/asr_chunks_v1/manifest.json"),
)
parser.add_argument(
    "--out", type=Path, default=Path("outputs/meeting_001_gigaam_vad_v1")
)
args = parser.parse_args()

MANIFEST = args.manifest
OUT = args.out
MODEL = "multilingual_ctc"

import gigaam
import torch


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

print("Загружаем GigaAM на CPU...", flush=True)
started = time.perf_counter()
model = gigaam.load_model(
    MODEL,
    device="cpu",
    fp16_encoder=False,
    download_root=str(Path.home() / ".cache" / "gigaam"),
)
model.eval()
load_seconds = time.perf_counter() - started

signature = inspect.signature(model.transcribe)
print(f"Параметры transcribe: {signature}", flush=True)

if "word_timestamps" not in signature.parameters:
    raise SystemExit(
        "В установленном transcribe нет word_timestamps. "
        "Ничего не обновляем автоматически: пришли этот вывод."
    )

OUT.mkdir(parents=True, exist_ok=True)
config = {
    "manifest": manifest,
    "model": MODEL,
    "device": "cpu",
    "fp16_encoder": False,
    "word_timestamps": True,
}
config_path = OUT / "run_config.json"

if config_path.exists():
    if read_json(config_path) != config:
        raise SystemExit("Настройки отличаются. Нужна новая папка результата.")
else:
    save_json(config_path, config)

all_words = []
chunk_records = []

for position, chunk in enumerate(manifest["chunks"], start=1):
    chunk_id = chunk["id"]
    audio = Path(chunk["audio_path"])
    response_path = OUT / f"chunk_{chunk_id:04d}.json"

    if not audio.is_file():
        raise SystemExit(f"Не найдено аудио: {audio}")

    print(
        f"[{position}/{len(manifest['chunks'])}] chunk_{chunk_id:04d}",
        flush=True,
    )

    if response_path.exists():
        response = read_json(response_path)
        print("  Используем сохранённый ответ", flush=True)
    else:
        started = time.perf_counter()
        with torch.inference_mode():
            result = model.transcribe(
                str(audio),
                word_timestamps=True,
            )
        elapsed = time.perf_counter() - started

        if not hasattr(result, "words"):
            raise SystemExit(
                f"Неожиданный результат: {type(result)}; {result!r}"
            )

        words = [
            {
                "word": item.text,
                "start": float(item.start),
                "end": float(item.end),
            }
            for item in result.words
        ]
        text = getattr(result, "text", None)
        if not isinstance(text, str):
            text = " ".join(item["word"] for item in words)

        response = {
            "model": MODEL,
            "text": text,
            "words": words,
            "transcribe_seconds": elapsed,
            "timing_scope": "local_transcribe_with_word_timestamps",
        }

    duration = chunk["duration_seconds"]
    offset = chunk["source_start_seconds"]

    if response["text"].strip() and not response["words"]:
        raise SystemExit(f"Есть текст, но нет слов: chunk_{chunk_id:04d}")

    for word_index, word in enumerate(response["words"]):
        start = float(word["start"])
        end = float(word["end"])

        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and 0 <= start <= end <= duration + 0.1
        ):
            raise SystemExit(
                f"Некорректные таймкоды в chunk_{chunk_id:04d}: {word}"
            )

        flags = list(chunk["review_flags"])
        if end > duration:
            flags.append("timestamp_exceeds_chunk_end")

        all_words.append({
            "word": word["word"],
            "start": offset + start,
            "end": offset + end,
            "local_start": start,
            "local_end": end,
            "chunk_id": chunk_id,
            "word_index_in_chunk": word_index,
            "review_flags": flags,
        })

    if not response_path.exists():
        save_json(response_path, response)

    chunk_records.append({
        "chunk_id": chunk_id,
        "response_path": response_path.as_posix(),
        "source_start_seconds": offset,
        "source_end_seconds": chunk["source_end_seconds"],
        "text": response["text"],
        "transcribe_seconds": response["transcribe_seconds"],
        "review_flags": chunk["review_flags"],
    })

    print(
        f"  Слов: {len(response['words'])}; "
        f"распознавание с таймкодами: "
        f"{response['transcribe_seconds']:.3f} с",
        flush=True,
    )

all_words.sort(
    key=lambda word: (
        word["start"], word["chunk_id"], word["word_index_in_chunk"]
    )
)

combined = {
    "source_audio": manifest["source_audio"],
    "model": MODEL,
    "manifest_path": MANIFEST.as_posix(),
    "time_reference": "original_audio_seconds",
    "manually_verified": False,
    "model_load_seconds_this_run": load_seconds,
    "chunks": chunk_records,
    "words": all_words,
    "statistics": {
        "chunks": len(chunk_records),
        "words": len(all_words),
        "transcribe_seconds_sum": sum(
            chunk["transcribe_seconds"] for chunk in chunk_records
        ),
        "timing_scope": "local_transcribe_with_word_timestamps",
    },
}

target = OUT / "transcript_original_time.json"
save_json(target, combined)

print(f"\nСохранено: {target}")
print(json.dumps(combined["statistics"], ensure_ascii=False, indent=2))
