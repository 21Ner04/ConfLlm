from pathlib import Path

from evals.evaluate_facts import score_facts
from evals.evaluate_rag import DEFAULT_DATASET, run_evaluation, score_retrieval


def test_fact_scorer_requires_source_and_expected_owner_deadline():
    expected = [{
        "id": "task",
        "kind": "task",
        "text_contains_any": ["checklist"],
        "source_utterance_ids": [4, 5],
        "owner": "speaker_3",
        "deadline": None,
    }]
    correct = [{
        "kind": "task",
        "text": "speaker_3 will prepare checklist",
        "owner": "speaker_3",
        "deadline": None,
        "evidence": [{"utterance_id": 4}, {"utterance_id": 5}],
    }]
    wrong_source = [{**correct[0], "evidence": [{"utterance_id": 4}]}]

    assert score_facts(expected, correct)["recall"] == 1.0
    assert score_facts(expected, wrong_source)["recall"] == 0.0


def test_rag_scorer_computes_recall_and_mrr_from_expected_fact_ids():
    cases = [
        {
            "id": "q1",
            "question": "who?",
            "reference_answer": "speaker 3",
            "relevant_fact_ids": [3],
            "results": [{"source_item_id": 8}, {"source_item_id": 3}],
        },
        {
            "id": "q2",
            "question": "when?",
            "reference_answer": "after check",
            "relevant_fact_ids": [2],
            "results": [{"source_item_id": 2}],
        },
    ]

    metrics = score_retrieval(cases, top_k=3)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mean_reciprocal_rank"] == 0.75
    assert metrics["queries"][0]["retrieved_fact_ids"] == ["8", "3"]


def test_versioned_rag_dataset_runs_end_to_end_without_external_embedding_service():
    def embed(text: str):
        value = text.casefold()
        return [
            float("пятниц" in value),
            float("безопасност" in value),
            float("список проверок" in value),
            float("дат" in value),
        ]

    report = run_evaluation(
        Path(DEFAULT_DATASET),
        top_k=3,
        embedding_fn=embed,
        chunking_fn=lambda text: [text],
    )

    assert report["dataset_version"] == "1.0.0"
    assert report["metrics"]["recall_at_k"] == 1.0
    assert report["metrics"]["mean_reciprocal_rank"] == 1.0
