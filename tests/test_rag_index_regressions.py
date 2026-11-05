import json

import pytest

from backend.rag_index import LocalVectorIndex


def _write_run(root, status):
    folder = root / "outputs" / "m1" / "run1"
    transcript_dir = folder / "transcript"
    transcript_dir.mkdir(parents=True)

    (transcript_dir / "aligned_transcript.json").write_text(
        json.dumps({
            "utterances": [{
                "id": 1,
                "speaker": "speaker_1",
                "start": 0,
                "end": 1,
                "text": "A completed meeting source.",
            }]
        }),
        encoding="utf-8",
    )

    status_path = folder / "pipeline_status.json"
    if status == "invalid_json":
        status_path.write_text("{broken", encoding="utf-8")
    elif status is not None:
        status_path.write_text(
            json.dumps({"status": status}),
            encoding="utf-8",
        )

    return folder


def _forbid_network(*args, **kwargs):
    raise AssertionError("Offline test attempted a network request")


@pytest.mark.parametrize(
    "status, expected_count",
    [
        ("failed", 0),
        ("running", 0),
        (None, 0),
        ("invalid_json", 0),
        ("completed", 1),
        ("completed_needs_review", 1),
    ],
)
def test_index_accepts_only_completed_runs_offline(
    tmp_path, monkeypatch, status, expected_count
):
    monkeypatch.setattr("urllib.request.urlopen", _forbid_network)
    _write_run(tmp_path, status)

    with LocalVectorIndex(
        tmp_path,
        embedding_fn=lambda text: [1.0],
        chunking_fn=lambda text: [text],
    ) as index:
        assert index.reindex() == expected_count
        assert index.count() == expected_count


def test_failed_reindex_preserves_previous_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _forbid_network)
    folder = _write_run(tmp_path, "completed_needs_review")

    with LocalVectorIndex(
        tmp_path,
        embedding_fn=lambda text: [1.0],
        chunking_fn=lambda text: [text],
    ) as index:
        assert index.reindex() == 1
        before = [
            tuple(row) for row in index._conn.execute(
                "SELECT * FROM rag_chunks ORDER BY id"
            ).fetchall()
        ]

        transcript_path = folder / "transcript/aligned_transcript.json"
        transcript_path.write_text(
            json.dumps({
                "utterances": [
                    {
                        "id": 1,
                        "speaker": "speaker_1",
                        "start": 0,
                        "end": 1,
                        "text": "New first passage.",
                    },
                    {
                        "id": 2,
                        "speaker": "speaker_2",
                        "start": 1,
                        "end": 2,
                        "text": "New second passage.",
                    },
                ]
            }),
            encoding="utf-8",
        )

        calls = 0

        def fail_on_second_embedding(text):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Simulated embedding failure")
            return [1.0]

        index.embedding_fn = fail_on_second_embedding

        with pytest.raises(RuntimeError, match="Simulated embedding failure"):
            index.reindex()

        after = [
            tuple(row) for row in index._conn.execute(
                "SELECT * FROM rag_chunks ORDER BY id"
            ).fetchall()
        ]

        assert calls == 2
        assert after == before
        assert index.count() == 1
