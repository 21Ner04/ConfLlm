import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def timestamp(seconds):
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def clean_text(text):
    return re.sub(r"\s+([,.;:!?…])", r"\1", text).strip()



def main():
    parser = argparse.ArgumentParser(
        description="Объединение нормализованного ASR, диаризации Nemotron и VAD."
    )
    parser.add_argument("--backend", required=True, choices=["gigaam", "parakeet"])
    parser.add_argument("--asr", type=Path)
    parser.add_argument(
        "--diarization", type=Path,
        default=Path("data/meetings/meeting_001/diarization/nemotron_response.json"),
    )
    parser.add_argument(
        "--vad", type=Path, default=Path("outputs/meeting_001_vad.json")
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--meeting-id", default="meeting_001")
    args = parser.parse_args()

    ASR_PATH = args.asr or Path(
        f"outputs/meeting_001_{args.backend}_vad_v1/transcript_original_time.json"
    )
    DIAR_PATH = args.diarization
    VAD_PATH = args.vad
    OUT = args.out or Path(
        f"data/meetings/meeting_001/transcript_{args.backend}_vad_v1"
    )
    MAX_GAP = 1.0

    if OUT.exists():
        raise SystemExit(f"Папка уже существует, не перезаписываем: {OUT}")

    asr = read_json(ASR_PATH)
    diar = read_json(DIAR_PATH)
    vad = read_json(VAD_PATH)

    if asr.get("time_reference") != "original_audio_seconds":
        raise ValueError("Ожидаются таймкоды исходной записи")

    if Path(asr["source_audio"]) != Path(vad["source"]):
        raise ValueError("ASR и VAD ссылаются на разные исходные файлы")

    diar_segments = diar["segments"]
    vad_segments = vad["segments"]
    units = []

    for index, word in enumerate(asr["words"]):
        start = float(word["start"])
        end = float(word["end"])
        text = word["word"].strip()

        if not text:
            continue
        if end < start:
            raise ValueError(f"Некорректные границы слова: {word}")

        zero_duration = end == start
        scores = defaultdict(float)

        for segment in diar_segments:
            left = float(segment["start"])
            right = float(segment["end"])
            score = (
                float(left <= start < right)
                if zero_duration else overlap(start, end, left, right)
            )
            if score > 0:
                scores[f"speaker_{segment['speaker']}"] += score

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        speaker = ranked[0][0] if ranked else "unknown"

        fraction = None
        if not zero_duration:
            covered = sum(
                overlap(start, end, float(s["start"]), float(s["end"]))
                for s in vad_segments
            )
            fraction = round(min(1.0, covered / (end - start)), 4)

        flags = list(word.get("review_flags", []))
        if not ranked:
            flags.append("no_speaker_overlap")
        if len(ranked) > 1:
            flags.append("multiple_speaker_candidates")
        if zero_duration:
            flags.append("zero_duration")
        if fraction == 0:
            flags.append("outside_vad")

        units.append({
            "id": len(units),
            "text": text,
            "start": start,
            "end": end,
            "speaker": speaker,
            "speaker_assignment": (
                "maximum_temporal_overlap" if ranked else "unassigned"
            ),
            "speaker_candidates": [
                {"speaker": name, "overlap": round(score, 6)}
                for name, score in ranked
            ],
            "chunk_id": word["chunk_id"],
            "source_word_index": index,
            "vad_speech_fraction": fraction,
            "review_flags": sorted(set(flags)),
        })

    utterances = []

    for unit in units:
        can_merge = (
            utterances
            and unit["speaker"] != "unknown"
            and utterances[-1]["speaker"] == unit["speaker"]
            and -0.5 <= unit["start"] - utterances[-1]["end"] <= MAX_GAP
        )

        if can_merge:
            current = utterances[-1]
            current["end"] = max(current["end"], unit["end"])
            current["text"] += " " + unit["text"]
            current["unit_indices"].append(unit["id"])
            current["chunk_ids"] = sorted(
                set(current["chunk_ids"]) | {unit["chunk_id"]}
            )
            current["review_flags"] = sorted(
                set(current["review_flags"]) | set(unit["review_flags"])
            )
        else:
            utterances.append({
                "id": len(utterances),
                "start": unit["start"],
                "end": unit["end"],
                "speaker": unit["speaker"],
                "text": unit["text"],
                "unit_indices": [unit["id"]],
                "chunk_ids": [unit["chunk_id"]],
                "review_flags": list(unit["review_flags"]),
            })

    for item in utterances:
        item["text"] = clean_text(item["text"])

    statistics = {
        "words": len(units),
        "utterances": len(utterances),
        "unknown_speaker_words": sum(
            unit["speaker"] == "unknown" for unit in units
        ),
        "multiple_speaker_candidate_words": sum(
            "multiple_speaker_candidates" in unit["review_flags"]
            for unit in units
        ),
        "outside_vad_words": sum(
            "outside_vad" in unit["review_flags"] for unit in units
        ),
        "flagged_words": sum(bool(unit["review_flags"]) for unit in units),
    }

    result = {
        "schema_version": "aligned_transcript_v1",
        "meeting_id": args.meeting_id,
        "source_audio": asr["source_audio"],
        "duration_seconds": vad["duration_seconds"],
        "model": asr["model"],
        "time_reference": "original_audio_seconds",
        "manually_verified": False,
        "sources": {
            "asr": ASR_PATH.as_posix(),
            "diarization": DIAR_PATH.as_posix(),
            "vad": VAD_PATH.as_posix(),
            "manifest": asr["manifest_path"],
        },
        "alignment": {
            "method": "maximum_temporal_overlap",
            "nearest_speaker_fallback": False,
            "max_utterance_gap_seconds": MAX_GAP,
        },
        "units": units,
        "utterances": utterances,
        "statistics": statistics,
    }

    lines = []
    for item in utterances:
        marker = " [CHECK]" if item["review_flags"] else ""
        lines.append(
            f"[{timestamp(item['start'])} — {timestamp(item['end'])}] "
            f"{item['speaker']}{marker}: {item['text']}"
        )

    OUT.mkdir(parents=True)
    (OUT / "aligned_transcript.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "aligned_transcript.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"JSON: {OUT / 'aligned_transcript.json'}")
    print(f"TXT:  {OUT / 'aligned_transcript.txt'}")
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print("\nПервые реплики:")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
