"""Run fact extraction on the versioned example and score it against gold labels."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(__file__).with_name("datasets") / "rag_quality_v1.json"


def score_facts(expected: list[dict], predicted: list[dict]) -> dict[str, Any]:
    unmatched = list(predicted)
    results = []
    true_positive = 0
    false_negative = 0

    for gold in expected:
        match_index = None
        errors = []
        for index, item in enumerate(unmatched):
            if item.get("kind") != gold["kind"]:
                continue
            evidence = {
                str(value) for value in (
                    item.get("source_utterance_ids")
                    or [entry.get("utterance_id") for entry in item.get("evidence", [])]
                ) if value is not None
            }
            expected_evidence = {str(value) for value in gold.get("source_utterance_ids", [])}
            if not expected_evidence.issubset(evidence):
                continue
            text = str(item.get("text") or "").casefold()
            keywords = [value.casefold() for value in gold.get("text_contains_any", [])]
            if keywords and not any(value in text for value in keywords):
                continue
            if "owner" in gold and item.get("owner") != gold["owner"]:
                continue
            if "deadline" in gold and item.get("deadline") != gold["deadline"]:
                continue
            match_index = index
            break
        if match_index is None:
            false_negative += 1
            errors.append("No extracted item matches kind, evidence, and expected fields")
        else:
            true_positive += 1
            unmatched.pop(match_index)
        results.append({"id": gold["id"], "passed": match_index is not None, "errors": errors})

    false_positive = len(unmatched)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        dataset_path = args.dataset.resolve()
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        corpus = dataset["corpus"]
        with tempfile.TemporaryDirectory(prefix="confllm-fact-eval-") as temporary:
            root = Path(temporary)
            transcript_path = root / "transcript.json"
            transcript_path.write_text(json.dumps({
                "meeting_id": corpus["meeting_id"],
                "utterances": corpus["utterances"],
            }, ensure_ascii=False), encoding="utf-8")
            out = root / "facts"
            command = [
                sys.executable,
                str(ROOT / "extract_meeting_facts.py"),
                "--source", str(transcript_path),
                "--out", str(out),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
            )
            facts_path = out / "facts.json"
            facts = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.is_file() else {}
            metrics = score_facts(dataset.get("fact_expectations", []), facts.get("items", []))
            stderr_tail = completed.stderr[-3000:]
            api_key = os.environ.get("LLM_API_KEY", "")
            if api_key:
                stderr_tail = stderr_tail.replace(api_key, "[REDACTED]")
            process_error = None if completed.returncode == 0 else {
                "return_code": completed.returncode,
                "stderr_tail": stderr_tail,
                "stdout_tail": completed.stdout[-1000:],
            }
    except Exception as error:
        print(f"Fact evaluation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    report = {
        "report_schema_version": "fact_evaluation_report_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset.get("dataset_id"),
        "dataset_version": dataset.get("dataset_version"),
        "model": os.environ.get("LLM_MODEL"),
        "prompt_version": facts.get("schema_version"),
        "metrics": {key: value for key, value in metrics.items() if key != "cases"},
        "cases": metrics["cases"],
        "pipeline_status": facts.get("status"),
        "failed_blocks": len(facts.get("failed_blocks", [])),
        "model_process_error": process_error,
        "notes": [
            "Labels are manually derived from the checked-in example transcript; this is a smoke benchmark, not a representative sample.",
            "Metric scores category, source-utterance coverage, and explicitly labeled owner/deadline fields; it does not judge paraphrase quality beyond the listed keywords.",
        ],
    }
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "benchmarks" / "evaluation_reports" / f"facts_{stamp}.json"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), **report["metrics"]}, ensure_ascii=False, indent=2))
    return 0 if metrics["recall"] == 1.0 and process_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
