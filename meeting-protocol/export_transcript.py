import json
from pathlib import Path

source = Path("outputs/meeting_001_transcript_v2.json")
target = Path("outputs/meeting_001_transcript.txt")

data = json.loads(source.read_text(encoding="utf-8"))


def timestamp(seconds):
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


lines = []
for segment in data["segments"]:
    start = timestamp(segment["start"])
    end = timestamp(segment["end"])
    text = segment["text"].strip()
    lines.append(f"[{start} — {end}] {text}")

target.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Сохранено: {target}; сегментов: {len(lines)}")