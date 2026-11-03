# Meeting Protocol

Сейчас реализованы два отдельных сценария:

1. Видео → аудио → распознавание речи → транскрипт с таймкодами.
2. Учебный текст → структурированный JSON протокола → Markdown.

Автоматическое объединение этих сценариев пока не реализовано.

## Как это работает

Модели запускаются на одном компьютере, а скрипты проекта обращаются к ним с другого по HTTP через IP-адрес и порт.

```text
FFmpeg
  Видео → WAV

whisper.cpp + Whisper
  WAV → JSON с текстом и таймкодами

export_transcript.py
  JSON → читаемый TXT с таймкодами

llama.cpp + Qwen
  Учебный текст → JSON протокола

export_markdown.py
  JSON протокола → Markdown
```

Whisper распознаёт речь. Qwen обрабатывает текст и формирует структуру протокола.

whisper.cpp и llama.cpp — отдельные программы для запуска соответствующих моделей.

## Что уже проверено

- Извлечение аудио из MP4 через FFmpeg.
- Отправка WAV на whisper-server по HTTP.
- Получение расшифровки в JSON.
- Экспорт сегментов в TXT с таймкодами.
- Обращение к Qwen через llama.cpp из Python.
- Генерация JSON протокола на учебном тексте.
- Экспорт учебного протокола в Markdown.

Диаризации и определения имён говорящих пока нет.

## Модели и программы

### Whisper

Текущая модель: `ggml-small.bin` — многоязычная Whisper small.

Использована готовая сборка whisper.cpp:
- релиз b5454;
- Windows x64;
- CUDA 12.4.0.

Текущие результаты относятся только к модели small. 

### Qwen

Qwen запускается через llama.cpp и доступен скриптам проекта по HTTP.

На учебном примере проверена цепочка:

```text
Текст → Qwen → JSON протокола → Markdown
```

Скрипты:
- `check_llm.py` — проверка обращения к модели;
- `test_protocol.py` — учебная генерация протокола;
- `export_markdown.py` — экспорт результата в Markdown.

Конкретный файл модели Qwen и команда запуска llama.cpp в этой инструкции пока не зафиксированы. Адрес сервера и используемые настройки нужно сверить с текущим кодом и локальным `.env`.

## Структура проекта

```text
meeting-protocol/
├── examples/
│   ├── meeting_001.txt
│   └── results/
├── data/meetings/meeting_001/
│   ├── source/video.mp4
│   └── audio/audio.wav
├── outputs/
├── check_llm.py
├── test_protocol.py
├── export_markdown.py
├── export_transcript.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

`examples/meeting_001.txt` — отдельный учебный текст, не расшифровка видео.

`data/` и `outputs/` хранят локальные материалы и результаты. Выбранные результаты, разрешённые к публикации, можно размещать в `examples/results/`.

## Подготовка окружения

Команды обработки выполняются из корня проекта в Ubuntu/WSL.

Системные зависимости:

```bash
sudo apt update
sudo apt install ffmpeg curl python3-venv
```

Python-окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Рабочие каталоги:

```bash
mkdir -p data/meetings/meeting_001/source
mkdir -p data/meetings/meeting_001/audio
mkdir -p outputs
```

## 1. Извлечение аудио

Поместите видео в:

```text
data/meetings/meeting_001/source/video.mp4
```

Выполните:

```bash
ffmpeg -i data/meetings/meeting_001/source/video.mp4 -map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le data/meetings/meeting_001/audio/audio.wav
```

Получится WAV: моно, 16 кГц, PCM 16 бит.

Видео остаётся без изменений, паузы не вырезаются. Перед распознаванием проверьте звук.

## 2. Запуск Whisper-сервера

На компьютере с моделями распакуйте whisper.cpp и скачайте `ggml-small.bin`.

Откройте PowerShell в папке с `whisper-server.exe`. Пример предполагает такую структуру:

```text
whisper/
├── whisper-server.exe
├── остальные файлы сборки
└── models/
    └── ggml-small.bin
```

Запустите:

```powershell
.\whisper-server.exe -m .\models\ggml-small.bin --host 0.0.0.0 --port 8081 -l ru
```

Оставьте сервер работающим. Проверьте сообщения об использовании CUDA/GPU.

`0.0.0.0` — адрес прослушивания. Для подключения с другого компьютера используйте реальный IP сервера.

Используйте сервер только в доверенной локальной сети. Не открывайте порт в интернет: аутентификация и HTTPS здесь не настроены.

## 3. Распознавание аудио

В терминале проекта задайте адрес компьютера с Whisper:

```bash
export WHISPER_HOST="100.100.1.100"
```

Замените пример на свой IP.

Отправьте WAV:

```bash
curl --fail-with-body "http://${WHISPER_HOST}:8081/inference" \
  -F "file=@data/meetings/meeting_001/audio/audio.wav" \
  -F "response_format=verbose_json" \
  -F "language=ru" \
  -F "split_on_word=true" \
  -o outputs/meeting_001_transcript_v2.json
```

Аудио передаётся автоматически в запросе. Ответ сохраняется в JSON.

`split_on_word=true` включает разбиение по словам, чтобы избежать разрывов внутри слова. Ошибки распознавания эта настройка не исправляет.

Проверка JSON:

```bash
python3 -m json.tool outputs/meeting_001_transcript_v2.json > /dev/null
```

## 4. Экспорт транскрипта

```bash
python3 export_transcript.py
```

Вход:

```text
outputs/meeting_001_transcript_v2.json
```

Выход:

```text
outputs/meeting_001_transcript.txt
```

Пример формата:

```text
[00:00:00 — 00:00:02] Текст сегмента
```

Пути пока заданы внутри скрипта. Исходный JSON сохраняется без изменений.

## 5. Учебный сценарий Qwen

Запустите настроенный HTTP-сервер llama.cpp с Qwen.

Проверьте в коде и локальном `.env` адрес сервера, пути входных файлов и параметры подключения.

С активированным Python-окружением:

```bash
python check_llm.py
python test_protocol.py
python export_markdown.py
```

Этот сценарий работает с учебным текстом. Реальная расшифровка Whisper пока не подключена к нему автоматически.

## Текущие результаты Whisper small

Использована многоязычная модель Whisper small — файл `ggml-small.bin`.

Одна и та же запись интервью длительностью около 10 минут 32 секунд обработана дважды:

| Файл в outputs/ | Модель | Настройки запуска | Результат |
|---|---|---|---|
| meeting_001_transcript.json | ggml-small.bin | Без явного включения split_on_word; русский язык задан на сервере | 258 сегментов; в проверенных фрагментах встречались разрывы внутри слов |
| meeting_001_transcript_v2.json | ggml-small.bin | split_on_word=true; language=ru также передан в запросе | 235 сегментов; в проверенных фрагментах разрывы внутри слов исчезли |

Оба JSON получены моделью small. Суффикс `v2` обозначает повторную обработку с изменёнными настройками, а не другую модель.

Из `meeting_001_transcript_v2.json` создан файл `meeting_001_transcript.txt` — читаемый транскрипт с временными границами.

При ручной сверке второго результата найдены:
- пропуски слов

Исходные JSON сохраняются без ручных исправлений. Полная точность по всей записи не измерена. Текст и таймкоды требуют проверки по исходному аудио.

### Сравнение моделей

Результатов medium и более крупных моделей пока нет. Текущие два JSON показывают различие настроек одной модели small, а не сравнение разных моделей.

При добавлении других моделей их результаты будут сохраняться отдельно с указанием файла модели и параметров запуска. Для сравнения будет использоваться та же запись и одинаковые параметры обработки.

## Публикация

Не добавляйте в репозиторий `.env`, ключи, модели и закрытые материалы.

Для демонстрации используйте выбранные файлы в `examples/results/`, если их разрешено распространять.

Перед коммитом проверяйте:

```bash
git status
git diff --cached --stat
git diff --cached
```
