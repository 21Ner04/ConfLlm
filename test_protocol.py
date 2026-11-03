import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict


class Protocol(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    release_friday_approved: bool
    revisit_date_after_security_check: bool
    security_check_owner: str | None
    checklist_owner: str | None
    checklist_deadline: str | None
    release_date_unresolved: bool


root = Path(__file__).resolve().parent
load_dotenv(root / ".env", override=True)
output = root / "outputs"
output.mkdir(exist_ok=True)

transcript = (root / "examples/meeting_001.txt").read_text(encoding="utf-8")

instructions = """
Извлеки факты из расшифровки. Верни только JSON с такими полями:
- release_friday_approved: boolean, утверждён ли выпуск именно в пятницу;
- revisit_date_after_security_check: boolean, договорились ли вернуться
  к обсуждению даты после проверки безопасности;
- security_check_owner: строка или null, явно назначенный исполнитель
  самой проверки безопасности;
- checklist_owner: строка или null, исполнитель подготовки списка проверок;
- checklist_deadline: строка или null, установленный срок подготовки списка;
- release_date_unresolved: boolean, осталась ли дата выпуска несогласованной.

Предложение не равно принятому решению.
Автор предложения не становится исполнителем автоматически.
Подготовка списка проверок и проведение проверки — разные задачи.
Неизвестные ответственные и сроки обозначай null.
"""

client = OpenAI(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
    timeout=120.0,
    max_retries=0,
)

print("Отправляем расшифровку на ПК...", flush=True)

response = client.chat.completions.create(
    model=os.environ["LLM_MODEL"],
    messages=[
        {"role": "system", "content": instructions},
        {"role": "user", "content": transcript},
    ],
    temperature=0,
    max_tokens=768,
    response_format={"type": "json_object"},
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

choice = response.choices[0]
raw = choice.message.content or ""
(output / "meeting_001_raw.txt").write_text(raw, encoding="utf-8")

print("Причина завершения:", choice.finish_reason)

if choice.finish_reason != "stop":
    raise SystemExit("Генерация не завершена нормально. Сырой ответ сохранён.")

try:
    protocol = Protocol.model_validate_json(raw)
except ValueError as error:
    print("Ошибка структуры:", error)
    raise SystemExit("Ответ сохранён в outputs/meeting_001_raw.txt")

actual = protocol.model_dump()
(output / "meeting_001_protocol.json").write_text(
    json.dumps(actual, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

expected = {
    "release_friday_approved": False,
    "revisit_date_after_security_check": True,
    "security_check_owner": None,
    "checklist_owner": "Виктор",
    "checklist_deadline": None,
    "release_date_unresolved": True,
}

print("\nПротокол:")
print(json.dumps(actual, ensure_ascii=False, indent=2))
print("\nПроверка фактов:")

passed = 0
for field, target in expected.items():
    ok = actual[field] == target
    passed += int(ok)
    print(f"{'PASS' if ok else 'FAIL'}: {field}")
    if not ok:
        print(f"  Ожидалось: {target!r}; получено: {actual[field]!r}")

print(f"\nИтог: {passed}/{len(expected)} проверок пройдено.")
