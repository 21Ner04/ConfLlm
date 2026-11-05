"""Evaluate retrieval against a small, versioned, human-labeled example set."""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(__file__).with_name("datasets") / "rag_quality_v1.json"


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def score_retrieval(cases: list[dict], top_k: int) -> dict[str, Any]:
    scored_cases = []
    total_relevant = 0
    total_found = 0
    reciprocal_ranks = []
    precisions = []

    for case in cases:
        relevant = {str(item_id) for item_id in case["relevant_fact_ids"]}
        retrieved = [str(item.get("source_item_id")) for item in case.get("results", [])[:top_k]]
        hits = [index for index, item_id in enumerate(retrieved, 1) if item_id in relevant]
        found = len(set(retrieved) & relevant)
        total_relevant += len(relevant)
        total_found += found
        reciprocal_ranks.append(1.0 / hits[0] if hits else 0.0)
        precisions.append(found / top_k if top_k else 0.0)
        scored_cases.append({
            "id": case["id"],
            "question": case["question"],
            "reference_answer": case["reference_answer"],
            "relevant_fact_ids": sorted(relevant),
            "retrieved_fact_ids": retrieved,
            "hits": found,
            "reciprocal_rank": round(reciprocal_ranks[-1], 6),
            "results": case.get("results", [])[:top_k],
        })

    count = len(cases)
    return {
        "recall_at_k": round(total_found / total_relevant, 6) if total_relevant else 0.0,
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / count, 6) if count else 0.0,
        "mean_precision_at_k": round(sum(precisions) / count, 6) if count else 0.0,
        "queries": scored_cases,
    }


def run_evaluation(
    dataset_path: Path,
    top_k: int,
    embedding_fn: Callable[[str], Sequence[float]] | None = None,
    chunking_fn: Callable[[str], Sequence[str]] | None = None,
) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset.get("dataset_version") != "1.0.0":
        raise ValueError("Unsupported evaluation dataset version")
    corpus = dataset["corpus"]

    with tempfile.TemporaryDirectory(prefix="confllm-rag-eval-") as temporary:
        root = Path(temporary)
        run_dir = root / "outputs" / corpus["meeting_id"] / corpus["run_id"]
        transcript_path = run_dir / "transcript" / "aligned_transcript.json"
        facts_path = run_dir / "facts" / "facts.json"
        transcript_path.parent.mkdir(parents=True)
        facts_path.parent.mkdir(parents=True)
        (run_dir / "pipeline_status.json").write_text(
            json.dumps({"status": "completed_needs_review"}),
            encoding="utf-8",
        )
        transcript_path.write_text(json.dumps({"utterances": corpus["utterances"]}, ensure_ascii=False), encoding="utf-8")
        facts_path.write_text(json.dumps({"items": corpus["facts"]}, ensure_ascii=False), encoding="utf-8")

        from backend.rag_index import LocalVectorIndex

        with LocalVectorIndex(
            root, embedding_fn=embedding_fn, chunking_fn=chunking_fn
        ) as index:
            indexed_chunks = index.reindex()
            retrieval_cases = []
            for case in dataset["queries"]:
                results = index.query(
                    case["question"],
                    top_k=top_k,
                    meeting_id=corpus["meeting_id"],
                    source_types=["proposal", "decision", "task", "open_question", "risk", "discussion"],
                )
                retrieval_cases.append({**case, "results": results})

    metrics = score_retrieval(retrieval_cases, top_k)
    return {
        "report_schema_version": "rag_evaluation_report_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["dataset_version"],
        "dataset_source": dataset.get("source"),
        "embedding_model": os.environ.get("LLM_EMBEDDING_MODEL") or os.environ.get("LLM_MODEL"),
        "top_k": top_k,
        "indexed_chunks": indexed_chunks,
        "metrics": {
            key: value for key, value in metrics.items() if key != "queries"
        },
        "queries": metrics["queries"],
        "notes": [
            "This evaluates retrieval of labeled evidence, not generated answer quality.",
            "The dataset is a small example-based smoke benchmark and is not representative of real meetings.",
            "Do not add private meeting content to a committed dataset without review and consent.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        report = run_evaluation(args.dataset.resolve(), args.top_k)
    except Exception as error:
        print(f"RAG evaluation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "benchmarks" / "evaluation_reports" / f"rag_{stamp}.json"
    if not output.is_absolute():
        output = ROOT / output
    _save_json(output, report)
    print(json.dumps({
        "report": str(output),
        "dataset": f"{report['dataset_id']}@{report['dataset_version']}",
        "indexed_chunks": report["indexed_chunks"],
        **report["metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
