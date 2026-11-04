import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CASES = [
    {
        "name": "opinion",
        "utterances": [
            ("speaker_1",
             "Я считаю, что через два года все разработчики будут работать "
             "только с агентами. Это мой личный прогноз, а не установленный факт.")
        ],
        "expected_kind": "discussion",
        "allowed_kinds": {"discussion"},
        "attribution": "speaker_1",
    },
    {
        "name": "advice",
        "utterances": [
            ("speaker_1",
             "Мой совет слушателям: попробуйте агента на небольшой задаче. "
             "Это рекомендация аудитории, я никому не назначаю поручение.")
        ],
        "expected_kind": "discussion",
        "allowed_kinds": {"discussion"},
    },
    {
        "name": "proposal",
        "utterances": [
            ("speaker_1", "Предлагаю перенести сервис на новую платформу."),
            ("speaker_2",
             "Пока не согласовано. Сначала обсудим риски. "
             "Сегодня решение о переносе не принимаем.")
        ],
        "expected_kind": "proposal",
        "allowed_kinds": {"proposal", "discussion", "risk", "open_question"},
    },
    {
        "name": "task",
        "utterances": [
            ("speaker_1",
             "Нужно подготовить отчёт о тестировании к пятнице. Кто возьмётся?"),
            ("speaker_2",
             "Я беру эту задачу: подготовлю отчёт о тестировании к пятнице."),
            ("speaker_1",
             "Договорились. Подготовку отчёта закрепляем за тобой, срок — к пятнице.")
        ],
        "expected_kind": "task",
        "allowed_kinds": {"task", "decision", "discussion"},
        "owner": "speaker_2",
        "deadline_contains": "пятниц",
    },
]


def main():
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = ROOT / "web_jobs" / "quality_checks" / session
    out.mkdir(parents=True)
    results = []

    for case in CASES:
        folder = out / case["name"]
        folder.mkdir()
        records = [
            {
                "id": index,
                "speaker": speaker,
                "start": float(index * 10),
                "end": float(index * 10 + 9),
                "text": text,
                "review_flags": [],
            }
            for index, (speaker, text) in enumerate(case["utterances"])
        ]
        source = folder / "transcript.json"
        source.write_text(
            json.dumps(
                {"meeting_id": case["name"], "utterances": records},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n=== Проверка: {case['name']} ===", flush=True)
        process = subprocess.run(
            [
                sys.executable,
                str(ROOT / "extract_meeting_facts.py"),
                "--source", str(source),
                "--out", str(folder / "facts"),
            ],
            cwd=ROOT,
            check=False,
        )

        errors = []
        facts_path = folder / "facts/facts.json"
        if process.returncode != 0 or not facts_path.is_file():
            errors.append("Извлечение не завершилось успешно")
        else:
            facts = json.loads(facts_path.read_text(encoding="utf-8"))
            items = facts["items"]
            selected = [
                item for item in items
                if item["kind"] == case["expected_kind"]
            ]
            if not selected:
                errors.append("Нет ожидаемой категории: " + case["expected_kind"])

            unexpected = {
                item["kind"] for item in items
            } - case["allowed_kinds"]
            if unexpected:
                errors.append("Лишние категории: " + ", ".join(sorted(unexpected)))

            if case.get("attribution"):
                if not selected or not all(
                    case["attribution"].casefold() in item["text"].casefold()
                    for item in selected
                ):
                    errors.append("В обсуждении не сохранён ID автора мнения")

            if case.get("owner"):
                matching = [
                    item for item in selected
                    if item.get("owner") == case["owner"]
                    and case["deadline_contains"] in
                    (item.get("deadline") or "").casefold()
                ]
                if not matching:
                    errors.append("Не сохранены исполнитель и срок задачи")

        passed = not errors
        results.append({
            "case": case["name"],
            "passed": passed,
            "errors": errors,
        })
        print("PASS" if passed else "FAIL: " + "; ".join(errors), flush=True)

    report = out / "summary.json"
    report.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nУспешно:", sum(item["passed"] for item in results), "/", len(results))
    print("Отчёт:", report)
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
