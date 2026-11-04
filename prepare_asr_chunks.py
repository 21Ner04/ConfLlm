import argparse
import json
import wave
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Подготовка WAV-фрагментов по VAD с сохранением исходного времени."
)
parser.add_argument(
    "--vad", type=Path, default=Path("outputs/meeting_001_vad.json")
)
parser.add_argument(
    "--out", type=Path,
    default=Path("data/meetings/meeting_001/asr_chunks_v1"),
)
parser.add_argument("--max-seconds", type=int, default=20)
args = parser.parse_args()

VAD_PATH = args.vad
OUT_DIR = args.out
MAX_SECONDS = args.max_seconds

if MAX_SECONDS <= 0:
    parser.error("--max-seconds должен быть положительным")

vad = json.loads(VAD_PATH.read_text(encoding="utf-8"))
source = Path(vad["source"])

if OUT_DIR.exists():
    raise SystemExit(
        f"Папка уже существует, ничего не перезаписано: {OUT_DIR}"
    )

with wave.open(str(source), "rb") as wav:
    params = wav.getparams()
    rate = wav.getframerate()
    total_frames = wav.getnframes()

    if (
        wav.getnchannels() != 1
        or wav.getsampwidth() != 2
        or rate != 16000
        or wav.getcomptype() != "NONE"
    ):
        raise ValueError("Ожидается WAV: mono, PCM 16-bit, 16000 Hz")

if vad["sample_rate"] != rate:
    raise ValueError("Частота VAD не совпадает с частотой WAV")

limit = int(MAX_SECONDS * rate)
pieces = []
previous_end = 0

for index, segment in enumerate(vad["segments"]):
    start = int(segment["start_sample"])
    end = int(segment["end_sample"])

    if not (0 <= start < end <= total_frames):
        raise ValueError(f"Некорректный VAD-интервал: {index}")

    if start < previous_end:
        raise ValueError("VAD-интервалы пересекаются или не отсортированы")

    previous_end = end
    forced = end - start > limit

    for left in range(start, end, limit):
        pieces.append({
            "start_sample": left,
            "end_sample": min(left + limit, end),
            "vad_segment_index": index,
            "forced_split": forced,
        })

if not pieces:
    raise SystemExit("В VAD нет интервалов речи. Файлы не созданы.")

groups = []

for piece in pieces:
    if (
        groups
        and piece["end_sample"] - groups[-1]["start_sample"] <= limit
    ):
        group = groups[-1]
        group["end_sample"] = piece["end_sample"]
        group["vad_segment_indices"].append(piece["vad_segment_index"])
        group["forced_split"] |= piece["forced_split"]
    else:
        groups.append({
            "start_sample": piece["start_sample"],
            "end_sample": piece["end_sample"],
            "vad_segment_indices": [piece["vad_segment_index"]],
            "forced_split": piece["forced_split"],
        })

OUT_DIR.mkdir(parents=True)
chunks = []

with wave.open(str(source), "rb") as wav:
    for index, group in enumerate(groups):
        start = group["start_sample"]
        end = group["end_sample"]
        target = OUT_DIR / f"chunk_{index:04d}.wav"

        wav.setpos(start)
        frames = wav.readframes(end - start)

        with wave.open(str(target), "wb") as output:
            output.setparams(params)
            output.writeframes(frames)

        chunks.append({
            "id": index,
            "audio_path": target.as_posix(),
            "source_start_sample": start,
            "source_end_sample": end,
            "source_start_seconds": start / rate,
            "source_end_seconds": end / rate,
            "duration_seconds": (end - start) / rate,
            "vad_segment_indices": sorted(
                set(group["vad_segment_indices"])
            ),
            "review_flags": (
                ["forced_split_inside_vad_interval"]
                if group["forced_split"] else []
            ),
        })

manifest = {
    "source_audio": source.as_posix(),
    "vad_path": VAD_PATH.as_posix(),
    "sample_rate": rate,
    "source_total_frames": total_frames,
    "max_chunk_seconds": MAX_SECONDS,
    "policy": "continuous_source_spans_grouped_from_vad",
    "silence_inside_chunks_preserved": True,
    "chunks": chunks,
}

manifest_path = OUT_DIR / "manifest.json"
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Создано фрагментов: {len(chunks)}")
print(f"Максимальная длительность: {max(c['duration_seconds'] for c in chunks):.3f} с")
print(f"Суммарная длительность: {sum(c['duration_seconds'] for c in chunks):.3f} с")
print(f"Фрагментов с принудительным разрезом: {sum(bool(c['review_flags']) for c in chunks)}")
print(f"Карта исходного времени: {manifest_path}")

for chunk in chunks[:5]:
    print(
        f"{chunk['id']:04d}: "
        f"{chunk['source_start_seconds']:.3f}–"
        f"{chunk['source_end_seconds']:.3f} с; "
        f"длина {chunk['duration_seconds']:.3f} с"
    )
