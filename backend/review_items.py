import copy
from datetime import datetime, timezone


def apply_items(current, updates, transcript):
    old = {item["id"]: item for item in current["items"]}
    by_id = {u["id"]: u for u in transcript["utterances"]}
    ids = [item["id"] for item in updates]
    if len(ids) != len(set(ids)):
        raise ValueError("Повторяющиеся ID пунктов")

    archived = copy.deepcopy(current.get("deleted_items", []))
    historical_ids = list(old) + [item["id"] for item in archived]
    next_id = max(
        current.get("next_item_id", 0),
        max(historical_ids, default=-1) + 1,
    )
    now = datetime.now(timezone.utc).isoformat()
    result = []

    for values in updates:
        values = dict(values)
        uid = values["id"]
        is_new = uid < 0
        if not is_new and uid not in old:
            raise ValueError("Неизвестный ID пункта")
        if not values["text"].strip():
            raise ValueError("Пустая формулировка")

        owner = values.get("owner")
        deadline = values.get("deadline")
        if owner is not None and (
            owner not in current["speakers"] or owner == "unknown"
        ):
            raise ValueError("Неизвестный исполнитель")
        if values["kind"] != "task" and (
            owner is not None or deadline is not None
        ):
            raise ValueError("Исполнитель и срок допустимы только для task")
        if deadline is not None and not deadline.strip():
            raise ValueError("Пустой срок")

        source_ids = values.pop("source_utterance_ids", None)
        if is_new and not source_ids:
            raise ValueError("Для нового пункта выбери реплику-источник")
        if source_ids is not None:
            source_ids = list(dict.fromkeys(source_ids))
            if not source_ids or any(uid not in by_id for uid in source_ids):
                raise ValueError("Проверь ID реплик-источников")

        if is_new:
            item = {
                "id": next_id,
                "original_text": "",
                "origin": "manual",
                "created_at": now,
                "evidence": [],
            }
            next_id += 1
        else:
            item = copy.deepcopy(old[uid])

        before = copy.deepcopy(item)
        for key in ["kind", "text", "owner", "deadline"]:
            item[key] = values[key]

        if source_ids is not None:
            item["evidence"] = [
                {
                    "utterance_id": source_id,
                    "quote": by_id[source_id]["text"],
                    "source_scope": "full_utterance",
                    "speaker": by_id[source_id]["speaker"],
                    "start": by_id[source_id]["start"],
                    "end": by_id[source_id]["end"],
                    "source_review_flags": by_id[source_id].get("review_flags", []),
                }
                for source_id in source_ids
            ]

        if is_new or item != before:
            item["edited"] = True
            item["edited_at"] = now
        result.append(item)

    kept = {item["id"] for item in result}
    for uid, item in old.items():
        if uid not in kept:
            deleted = copy.deepcopy(item)
            deleted["deleted_at"] = now
            archived.append(deleted)

    return result, archived, next_id
