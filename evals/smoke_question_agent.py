"""Live smoke evaluation of the question agent on isolated fixtures."""

import json
import re
import tempfile
from pathlib import Path

from backend import main
from backend.memory_store import MemoryStore
from backend.question_discovery import (
    discover_questions,
    list_confirmed_questions,
    review_question_suggestion,
)
from backend.rag_index import LocalVectorIndex


LOCAL_QUESTION = "Какой формат встречи согласован?"
DATE_QUESTION = "На какую дату согласован запуск пилота проекта «Север»?"
BUDGET_QUESTION = "Какой точный бюджет в рублях утверждён для пилота проекта «Север»?"
REVENUE_QUESTION = "Какой точный размер годовой выручки компании «Орион» в рублях?"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def utterance(identifier, text):
    return {
        "id": identifier,
        "text": text,
        "speaker": "speaker_1" if identifier % 2 else "speaker_2",
        "start": float(identifier * 10),
        "end": float(identifier * 10 + 8),
    }


def meeting(root, meeting_id, run_id, utterances):
    folder = root / "outputs" / meeting_id / run_id
    write_json(folder / "pipeline_status.json", {"status": "completed"})
    write_json(
        folder / "transcript" / "aligned_transcript.json",
        {"utterances": utterances},
    )
    write_json(folder / "facts" / "facts.json", {"items": []})



def check_budget_result(agent):
    status = agent["status"]
    if status == "insufficient_sources":
        return (
            ["У отказа не должно быть источников итогового ответа."]
            if agent.get("sources") else []
        )

    if status != "answered":
        return ["Не получен проверяемый результат по бюджету."]

    answer = agent.get("answer", "").casefold()
    if "не утвержд" not in answer and "не согласован" not in answer:
        return ["Нет явного сообщения, что бюджет не утверждён."]

    # Remove only application-generated numeric citation markers.
    without_citations = re.sub(r"\[\d+\]", "", agent.get("answer", ""))
    if re.search(r"\d", without_citations):
        return ["В ответ о неутверждённом бюджете добавлено числовое значение."]

    supported = any(
        source.get("meeting_id") == "demo_original"
        and str(source.get("utterance_id")) == "6"
        and source.get("text") == (
            "Бюджет не утверждён. Точной суммы в рублях нет."
        )
        for source in agent.get("sources", [])
    )
    if not supported:
        return ["Нет прямого источника сообщения о неутверждённом бюджете."]
    return []


def check_missing_data_result(agent):
    failures = []
    if agent["status"] != "insufficient_sources":
        failures.append(
            "Для отсутствующих сведений о выручке ожидался insufficient_sources."
        )
    if agent.get("sources"):
        failures.append(
            "У отказа по отсутствующей выручке есть источники итогового ответа."
        )
    return failures


def main_smoke():
    original_root = main.ROOT
    failures = []

    try:
        with tempfile.TemporaryDirectory(
            prefix="confllm-agent-release-smoke-"
        ) as temporary:
            root = Path(temporary)
            main.ROOT = root

            meeting(
                root, "demo_original", "20261101T090000_000000Z",
                [
                    utterance(1, LOCAL_QUESTION),
                    utterance(
                        2,
                        "Согласована онлайн-встреча без очного участия.",
                    ),
                    utterance(3, DATE_QUESTION),
                    utterance(4, "Дата запуска пока не определена."),
                    utterance(5, BUDGET_QUESTION),
                    utterance(
                        6,
                        "Бюджет не утверждён. Точной суммы в рублях нет.",
                    ),
                    utterance(7, REVENUE_QUESTION),
                ],
            )
            meeting(
                root, "demo_followup", "20261105T090000_000000Z",
                [
                    utterance(
                        10,
                        "Согласовали запуск пилота проекта «Север» "
                        "на 20 ноября. Это утверждённая дата запуска.",
                    ),
                ],
            )

            # Fixture setup, not a test of live question extraction.
            discovered = discover_questions(
                root,
                "demo_original",
                "20261101T090000_000000Z",
                extract=lambda context: {"questions": [
                    {"text": LOCAL_QUESTION, "source_ids": [1]},
                    {"text": DATE_QUESTION, "source_ids": [3]},
                    {"text": BUDGET_QUESTION, "source_ids": [5]},
                    {"text": REVENUE_QUESTION, "source_ids": [7]},
                ]},
            )

            for suggestion in discovered["items"]:
                review_question_suggestion(
                    root, suggestion["suggestion_id"], "accepted"
                )

            questions = {
                item["text"]: item
                for item in list_confirmed_questions(root)
            }
            expected = {
                LOCAL_QUESTION, DATE_QUESTION,
                BUDGET_QUESTION, REVENUE_QUESTION,
            }
            if set(questions) != expected:
                raise RuntimeError("Не созданы все учебные вопросы.")

            print(
                "Строю временный индекс с настоящими embeddings и chunking...",
                flush=True,
            )
            with LocalVectorIndex(root) as index:
                print(f"Фрагментов: {index.reindex()}", flush=True)

            cases = [
                ("Исходный диалог", LOCAL_QUESTION, "source_dialogue"),
                ("Ответ в другой встрече", DATE_QUESTION, "all_meetings"),
                ("Неутверждённый бюджет", BUDGET_QUESTION, "all_meetings"),
                ("Нет сведений о выручке", REVENUE_QUESTION, "all_meetings"),
            ]

            for label, text, scope in cases:
                print(f"\n=== {label} ===", flush=True)

                response = main.run_question_agent_api(
                    main.QuestionAgentRunRequest(
                        question_id=questions[text]["question_id"],
                        answer_scope=scope,
                    )
                )
                agent = response["agent"]

                print(json.dumps({
                    "question": text,
                    "scope": scope,
                    "status": agent["status"],
                    "stop_reason": agent["stop_reason"],
                    "answer": agent["answer"],
                    "sources": agent["sources"],
                    "agent_trace": agent["agent_trace"],
                    "claim_checks": agent.get("claim_checks"),
                    "metrics": agent.get("metrics"),
                }, ensure_ascii=False, indent=2), flush=True)

                if scope == "source_dialogue":
                    if agent["status"] != "answered":
                        failures.append("Нет ответа по исходному диалогу.")
                    if "онлайн" not in agent["answer"].casefold():
                        failures.append("Не указан согласованный онлайн-формат.")
                    if not any(
                        str(source.get("utterance_id")) == "2"
                        and source.get("meeting_id") == "demo_original"
                        for source in agent["sources"]
                    ):
                        failures.append("Нет цитаты прямого ответа о формате.")
                    if any(
                        source.get("meeting_id") != "demo_original"
                        for source in agent["sources"]
                    ):
                        failures.append("Режим диалога использовал другую встречу.")

                elif text == DATE_QUESTION:
                    if agent["status"] != "answered":
                        failures.append("Нет ответа из другой встречи.")
                    if "20 ноября" not in agent["answer"].casefold():
                        failures.append("Нет ожидаемой даты.")
                    if not any(
                        source.get("meeting_id") == "demo_followup"
                        for source in agent["sources"]
                    ):
                        failures.append("Нет источника утверждённой даты.")

                elif text == BUDGET_QUESTION:
                    failures.extend(check_budget_result(agent))
                else:
                    failures.extend(check_missing_data_result(agent))

            # Real synchronization must not remove separately confirmed questions.
            with MemoryStore(root) as memory:
                memory.sync()

            after = list_confirmed_questions(root)
            if len(after) != 4 or any(
                item["status"] != "open" for item in after
            ):
                failures.append(
                    "Подтверждённые вопросы исчезли или изменили статус."
                )

    finally:
        main.ROOT = original_root
        print(
            "\nВременное окружение закрыто. "
            "Рабочие встречи и базы не использовались.",
            flush=True,
        )

    if failures:
        print("\nПРОВЕРКА НЕ ПРОЙДЕНА:", flush=True)
        for failure in failures:
            print("- " + failure, flush=True)
        raise SystemExit(1)

    print(
        "\nПРОВЕРКА ПРОЙДЕНА: исходный диалог, ответ в другой встрече, "
        "неутверждённый бюджет без выдуманной суммы, "
        "отказ по отсутствующей выручке и сохранение статусов.",
        flush=True,
    )


if __name__ == "__main__":
    main_smoke()
