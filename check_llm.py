import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"), override=True)

client = OpenAI(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
    timeout=120.0,
    max_retries=0,
)

print("Отправляем запрос на ПК...", flush=True)

response = client.chat.completions.create(
    model=os.environ["LLM_MODEL"],
    messages=[
        {
            "role": "user",
            "content": "Ответь одним коротким предложением: что такое протокол встречи?",
        }
    ],
    max_tokens=512,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

print(response.choices[0].message.content)

choice = response.choices[0]
print("Причина завершения:", choice.finish_reason)
print("Текст ответа:", repr(choice.message.content))
print("Поля сообщения:", choice.message.model_dump())
print("Использованные токены:", response.usage)
