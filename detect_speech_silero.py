import argparse
import json
import time
import wave
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Определение интервалов речи через Silero VAD."
)
parser.add_argument(
    "--source", type=Path,
    default=Path("data/meetings/meeting_001/audio/audio.wav"),
)
parser.add_argument(
    "--out", type=Path,
    default=Path("outputs/meeting_001_vad.json"),
)
args = parser.parse_args()

SOURCE = args.source
TARGET = args.out
SAMPLE_RATE = 16000

if not SOURCE.is_file():
    parser.error(f"Не найден WAV: {SOURCE}")
if SOURCE.resolve() == TARGET.resolve():
    parser.error("Выходной файл не должен совпадать с исходным WAV")

import torch
from silero_vad import load_silero_vad, get_speech_timestamps

torch.set_num_threads(1)

with wave.open(str(SOURCE), "rb") as wav:
    if (
        wav.getnchannels() != 1
        or wav.getsampwidth() != 2
        or wav.getframerate() != SAMPLE_RATE
        or wav.getcomptype() != "NONE"
    ):
        raise ValueError("Ожидается WAV: mono, PCM 16-bit, 16000 Hz")

    samples = wav.readframes(wav.getnframes())

audio = (
    torch.frombuffer(bytearray(samples), dtype=torch.int16)
    .to(torch.float32)
    / 32768.0
)

parameters = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30,
}

model = load_silero_vad()

started = time.perf_counter()
intervals = get_speech_timestamps(
    audio,
    model,
    sampling_rate=SAMPLE_RATE,
    return_seconds=False,
    **parameters,
)
processing_seconds = time.perf_counter() - started

segments = [
    {
        "id": index,
        "start_sample": int(item["start"]),
        "end_sample": int(item["end"]),
        "start": item["start"] / SAMPLE_RATE,
        "end": item["end"] / SAMPLE_RATE,
    }
    for index, item in enumerate(intervals)
]

duration = audio.numel() / SAMPLE_RATE
speech_duration = sum(s["end"] - s["start"] for s in segments)

result = {
    "source": SOURCE.as_posix(),
    "model": "silero_vad",
    "device": "cpu",
    "sample_rate": SAMPLE_RATE,
    "duration_seconds": duration,
    "parameters": parameters,
    "processing_seconds": processing_seconds,
    "speech_seconds": speech_duration,
    "non_speech_seconds": max(0.0, duration - speech_duration),
    "segments": segments,
}

TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Сохранено: {TARGET}")
print(f"Интервалов речи: {len(segments)}")
print(f"Длительность записи: {duration:.2f} с")
print(f"Интервалы, отмеченные как речь: {speech_duration:.2f} с")
print(f"Время VAD без загрузки модели: {processing_seconds:.2f} с")

for segment in segments[:10]:
    print(f'{segment["start"]:.3f} — {segment["end"]:.3f}')
