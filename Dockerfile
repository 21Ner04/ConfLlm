FROM python:3.12-slim-bookworm

ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg curl fonts-dejavu-core libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-export.txt web_requirements.txt \
     requirements-torch.txt requirements-vad.txt ./

RUN python -m pip install --no-cache-dir \
      -r requirements-torch.txt \
      --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir \
      -r requirements.txt \
      -r requirements-export.txt \
      -r web_requirements.txt \
      -r requirements-vad.txt \
      -c requirements-torch.txt \
    && python -m pip check \
    && python -c "import torch; from silero_vad import load_silero_vad; load_silero_vad(); print('VAD import OK')"

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
       --create-home --shell /bin/sh app \
    && mkdir -p /app/data /app/outputs /app/web_jobs \
    && chown -R app:app /app

COPY --chown=app:app *.py ./
COPY --chown=app:app backend ./backend
COPY --chown=app:app evals ./evals

USER app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
