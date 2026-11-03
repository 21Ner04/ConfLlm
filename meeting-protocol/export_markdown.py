import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = json.loads(
    (root / "outputs/meeting_001_protocol.json").read_text(encoding="utf-8")
)

decisions = []
if data["revisit_date_after_security_check"]:
    decisions.append(
        "- Вернуться к обсуждению даты выпуска после проверки безопасности."
    )
if data["release_friday_approved"]:
    decisions.append("- Выпуск продукта в пятницу утверждён.")

tasks = []
if data["checklist_owner"] is not None:
    deadline = data["checklist_deadline"] or "не указан"
    tasks.append(
        f"- Подготовить список проверок.\n"
        f"  - Ответственный: {data['checklist_owner']}.\n"
        f"  - Срок: {deadline}."
    )

questions = []
if data["release_date_unresolved"]:
    questions.append("- Окончательная дата выпуска.")
if data["security_check_owner"] is None:
    questions.append("- Кто проведёт проверку безопасности?")

text = "\n".join([
    "# Протокол учебной встречи",
    "",
    "Документ сформирован из результата тестового извлечения фактов.",
    "",
    "## Решения",
    "",
    "\n".join(decisions) or "- Не зафиксированы.",
    "",
    "## Поручения",
    "",
    "\n".join(tasks) or "- Не зафиксированы.",
    "",
    "## Открытые вопросы и необходимые уточнения",
    "",
    "\n".join(questions) or "- Не зафиксированы.",
    "",
])

path = root / "outputs/meeting_001_protocol.md"
path.write_text(text, encoding="utf-8")
print(f"Документ сохранён: {path.name}")
