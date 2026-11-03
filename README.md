# Meeting Protocol

Сейчас реализованы два отдельных сценария:

1. Видео → аудио → распознавание речи, VAD и диаризация → транскрипт с таймкодами и говорящими.
2. Учебный текст → структурированный JSON протокола → Markdown.

Автоматическая генерация протокола из реальной расшифровки пока не реализована.

## Как это работает

Модели запускаются на ПК, скрипты проекта выполняются на ноутбуке в Ubuntu/WSL и обращаются к серверам по HTTP. Silero VAD выполняется локально на CPU ноутбука.

```text
FFmpeg
  Видео → WAV

whisper.cpp + Whisper
  WAV → JSON с текстом и таймкодами

Silero VAD
  WAV → интервалы речи

NeMo-Speech.cpp + Nemotron 3 Diarization
  WAV → интервалы говорящих

merge_transcript.py
  Whisper JSON + Nemotron JSON + VAD JSON
    → объединённый JSON и TXT: время, говорящий, текст

export_transcript.py
  Whisper JSON → TXT с таймкодами без говорящих

llama.cpp + Qwen
  Учебный текст → JSON протокола

export_markdown.py
  JSON протокола → Markdown
```

Whisper распознаёт речь. Silero VAD определяет участки речи. Nemotron определяет интервалы говорящих. Qwen обрабатывает текст и формирует структуру протокола.

## Что уже проверено

- Извлечение аудио из MP4 через FFmpeg.
- Отправка WAV на whisper-server по HTTP.
- Получение расшифровки в JSON.
- Экспорт сегментов в TXT с таймкодами.
- Определение интервалов речи через Silero VAD на CPU.
- Диаризация через Nemotron на ПК с использованием Vulkan.
- Обращение к серверу NeMo-Speech.cpp из WSL.
- Объединение текста с интервалами говорящих.
- Предварительная ручная проверка трёх фрагментов объединённого транскрипта.
- Обращение к Qwen через llama.cpp из Python.
- Генерация JSON протокола на учебном тексте.
- Экспорт учебного протокола в Markdown.

Говорящие обозначены как `speaker_1` и `speaker_2`. Определение их имён не реализовано.

## Модели и программы

### Whisper

Текущая модель: `ggml-small.bin` — многоязычная Whisper small.

Использована готовая сборка whisper.cpp:
- релиз b5454;
- Windows x64;
- CUDA 12.4.0.

Текущие результаты относятся только к модели small.

### Silero VAD

Запускается через Python-библиотеку `silero_vad` на CPU ноутбука.

Скрипт: `run_vad.py`.

Параметры:

```text
threshold: 0.5
min_speech_duration_ms: 250
min_silence_duration_ms: 100
speech_pad_ms: 30
```

Скрипт сохраняет границы речи в отсчётах и секундах. Исходное аудио не изменяет, тишину не вырезает.

### Nemotron 3 Diarization

Программа: NeMo-Speech.cpp v0.3.0.

Модель: `Nemotron-3-Diarization.q8_0.gguf`.

Рабочая конфигурация:
- Windows x64;
- RTX 2070;
- Vulkan-сборка;
- устройство сервера: `vulkan:0`.

CUDA-сборка в текущем окружении завершалась ошибкой инициализации CUDA. Рабочий запуск выполнен через Vulkan.

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
│   ├── audio/audio.wav
│   ├── vad_check/
│   ├── diarization/
│   │   └── nemotron_response.json
│   ├── transcript/
│   │   ├── aligned_transcript.json
│   │   └── aligned_transcript.txt
│   └── alignment_check/
├── outputs/
│   ├── meeting_001_transcript.json
│   ├── meeting_001_transcript_v2.json
│   ├── meeting_001_transcript.txt
│   └── meeting_001_vad.json
├── check_llm.py
├── test_protocol.py
├── export_markdown.py
├── export_transcript.py
├── run_vad.py
├── merge_transcript.py
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

Для `run_vad.py` необходимы `torch` и `silero-vad`. Для `merge_transcript.py` достаточно стандартной библиотеки Python.

Рабочие каталоги:

```bash
mkdir -p data/meetings/meeting_001/source
mkdir -p data/meetings/meeting_001/audio
mkdir -p data/meetings/meeting_001/diarization
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

На ПК с моделями распакуйте whisper.cpp и скачайте `ggml-small.bin`.

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

`0.0.0.0` — адрес прослушивания. Для подключения с ноутбука используйте реальный IP ПК.

Используйте сервер только в доверенной локальной сети. Не открывайте порт в интернет: аутентификация и HTTPS здесь не настроены.

## 3. Распознавание аудио

В терминале проекта задайте адрес ПК с Whisper:

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

`split_on_word=true` используется для предотвращения разрывов слов между сегментами. Элементы массива `words` при этом могут оставаться частями слов. Ошибки распознавания эта настройка не исправляет.

Проверка JSON:

```bash
python3 -m json.tool outputs/meeting_001_transcript_v2.json > /dev/null
```

## 4. Экспорт транскрипта без говорящих

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

Пути заданы внутри скрипта. Исходный JSON сохраняется без изменений.

## 5. Определение интервалов речи

С активированным Python-окружением:

```bash
python3 run_vad.py
```

Вход:

```text
data/meetings/meeting_001/audio/audio.wav
```

Выход:

```text
outputs/meeting_001_vad.json
```

Для текущей записи получено:

| Показатель | Результат |
|---|---:|
| Длительность аудио | 632,30 с |
| Интервалы речи | 125 |
| Отмечено как речь | 593,45 с |
| Вне интервалов речи | 38,85 с |
| Время обработки без загрузки модели | 13,90 с |

Это результат одного запуска. VAD не определяет говорящих или текст и не удаляет паузы из исходного файла.

## 6. Запуск сервера диаризации

На ПК распакуйте Vulkan-сборку NeMo-Speech.cpp v0.3.0.

Расположение файлов:

```text
nemotron/
├── bin/
│   ├── nemo-speech.exe
│   └── остальные файлы сборки
└── models/
    └── Nemotron-3-Diarization.q8_0.gguf
```

Откройте PowerShell в `bin`.

Проверьте устройства:

```powershell
.\nemo-speech.exe doctor
```

Запустите сервер:

```powershell
.\nemo-speech.exe serve --host 0.0.0.0 --port 8090 --device vulkan:0 --diar-model "..\models\Nemotron-3-Diarization.q8_0.gguf"
```

Проверьте, что `vulkan:0` соответствует нужной видеокарте. Оставьте сервер работающим.

Используйте сервер только в доверенной локальной сети. В этой команде API-ключ и HTTPS не настроены.

## 7. Диаризация аудио

На ноутбуке в терминале WSL:

```bash
export NEMOTRON_URL="http://100.100.1.100:8090"
```

Замените IP на адрес ПК.

Проверка подключения:

```bash
curl --connect-timeout 5 -sS "$NEMOTRON_URL/v1/models"
```

В рабочем запуске сервер возвращает возможность `diarization` и устройство `vulkan:0`.

Отправьте аудио:

```bash
mkdir -p data/meetings/meeting_001/diarization

curl --connect-timeout 10 \
  --fail-with-body \
  --show-error \
  -F "file=@data/meetings/meeting_001/audio/audio.wav" \
  "$NEMOTRON_URL/v1/audio/diarizations" \
  -o data/meetings/meeting_001/diarization/nemotron_response.json
```

Ответ содержит интервалы:

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 1.519,
      "speaker": 1
    }
  ]
}
```

В текущей записи выделены два идентификатора говорящих. HTTP-запрос завершился примерно за 18 секунд, включая передачу аудио и ожидание результата. Отдельное время вычисления модели не измерено.

## 8. Объединение текста и говорящих

```bash
python3 merge_transcript.py
```

Входы:
- `outputs/meeting_001_transcript_v2.json`;
- `outputs/meeting_001_vad.json`;
- `data/meetings/meeting_001/diarization/nemotron_response.json`.

Выходы:
- `data/meetings/meeting_001/transcript/aligned_transcript.json`;
- `data/meetings/meeting_001/transcript/aligned_transcript.txt`.

Скрипт:
- собирает текстовые элементы из частей `words`;
- назначает говорящего по максимальному временному пересечению;
- при отсутствии пересечения допускает назначение ближайшего говорящего в пределах 0,15 с с проверкой конкурирующего кандидата;
- отмечает приблизительные назначения и спорные интервалы;
- объединяет соседние элементы одного говорящего в реплики;
- сохраняет VAD отдельным разделом, не удаляя текст.

Пример TXT:

```text
[00:00:00.090 — 00:00:01.280] speaker_1 [CHECK]: Лёша, привет.
[00:00:01.530 — 00:00:02.170] speaker_2: Здорово.
```

`[CHECK]` обозначает необходимость проверки, а не доказанную ошибку.

Текущий результат:
- 1602 текстовых элемента;
- 48 реплик;
- 2 элемента без назначения говорящего: смех и отдельное тире;
- 204 элемента с предупреждениями.

Проведена предварительная ручная проверка участков 0–30, 284–294 и 585–600 с. Полная запись не проверена; `manually_verified` остаётся `false`.

Ограничения:
- часть ASR-элементов имеет нулевую длительность;
- назначения возле границ приблизительные;
- пересечения голосов требуют проверки;
- имена участников не определяются;
- VAD-нарезка для отправки в ASR пока не интегрирована.

## 9. Учебный сценарий Qwen

Запустите настроенный HTTP-сервер llama.cpp с Qwen.

Проверьте в коде и локальном `.env` адрес сервера, пути входных файлов и параметры подключения.

С активированным Python-окружением:

```bash
python check_llm.py
python test_protocol.py
python export_markdown.py
```

Этот сценарий работает с учебным текстом. Объединённый транскрипт реальной записи пока не подключён к нему автоматически.

## Текущие результаты Whisper small

Использована многоязычная модель Whisper small — файл `ggml-small.bin`.

Одна и та же запись интервью длительностью около 10 минут 32 секунд обработана дважды:

| Файл в outputs/ | Модель | Настройки | Результат |
|---|---|---|---|
| meeting_001_transcript.json | ggml-small.bin | Без явного включения split_on_word; русский язык задан на сервере | 258 сегментов; встречались разрывы внутри слов |
| meeting_001_transcript_v2.json | ggml-small.bin | split_on_word=true; language=ru передан в запросе | 235 сегментов; в проверенных фрагментах разрывы внутри слов между сегментами исчезли |

Оба JSON получены моделью small. Суффикс `v2` обозначает повторную обработку с изменёнными настройками, а не другую модель.

При ручной сверке второго результата найдены пропуски слов.

Исходные JSON сохраняются без ручных исправлений. Полная точность по всей записи не измерена. Текст и таймкоды требуют проверки по исходному аудио.

### Сравнение моделей

GigaAM Multilingual, Parakeet и другие размеры Whisper пока не сравнивались.

Для сравнения нужны:
- вручную проверенный эталон;
- одинаковые входные данные и условия обработки;
- WER и CER;
- время обработки и метрика скорости;
- при возможности — потребление VRAM.

## Что осталось сделать

- Интегрировать VAD-подготовку фрагментов в распознавание с сохранением исходных временных смещений.
- Провести сравнение ASR-моделей.
- Подключить объединённый транскрипт к генерации протокола.
- Добавить экспорт DOCX и PDF.
- Сделать веб-интерфейс полного цикла.
- Подготовить Docker Compose, `.env.example` и GitHub Actions.

## Публикация

Не добавляйте в репозиторий `.env`, ключи, модели и закрытые материалы.

Локальные материалы и результаты должны исключаться через `.gitignore`:

```gitignore
/data/
/outputs/
*.bak
```

Уже отслеживаемые файлы нужно отдельно убрать из индекса Git. Добавление правила в `.gitignore` не удаляет их из истории коммитов.

Для демонстрации используйте выбранные файлы в `examples/results/`, если их разрешено распространять.

Перед коммитом проверяйте:

```bash
git status
git diff --cached --stat
git diff --cached
```