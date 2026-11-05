import json
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.quality_report import collect_quality_report


def test_quality_report_summarizes_stage_errors_and_coverage_without_claiming_accuracy():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "outputs"
        work = root / "work"
        (output / "transcript").mkdir(parents=True)
        (output / "facts").mkdir()
        (work / "asr_chunks").mkdir(parents=True)
        (output / "vad.json").write_text(json.dumps({
            "duration_seconds": 20,
            "speech_seconds": 12,
            "segments": [{"start": 1}, {"start": 5}],
        }), encoding="utf-8")
        (work / "asr_chunks" / "manifest.json").write_text(json.dumps({
            "chunks": [{"duration_seconds": 10, "review_flags": ["forced_split_inside_vad_interval"]}],
        }), encoding="utf-8")
        (output / "parakeet").mkdir()
        (output / "parakeet" / "transcript_original_time.json").write_text(json.dumps({"words": [{}, {}], "model": "test-model"}), encoding="utf-8")
        (output / "transcript" / "aligned_transcript.json").write_text(json.dumps({
            "utterances": [
                {"speaker": "unknown", "review_flags": ["no_speaker_overlap"]},
                {"speaker": "speaker_1", "review_flags": []},
            ]
        }), encoding="utf-8")
        (output / "facts" / "facts.json").write_text(json.dumps({
            "items": [{"kind": "task"}],
            "failed_blocks": [{"block_id": 0}],
            "warnings": [],
        }), encoding="utf-8")
        stages = [{"name": "Parakeet", "status": "failed", "error_type": "CalledProcessError"}]

        report = collect_quality_report(output, work, stages)

        assert report["chunking"]["forced_split_count"] == 1
        assert report["transcript"]["unknown_speaker_count"] == 1
        assert report["fact_extraction"]["failed_block_count"] == 1
        assert report["stages"][0]["status"] == "failed"
        assert "not an accuracy score" in report["interpretation"]
