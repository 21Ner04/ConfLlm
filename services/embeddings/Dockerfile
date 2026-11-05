FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /service

RUN python -m pip install --no-cache-dir torch \
      --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir \
      sentence-transformers fastapi uvicorn \
    && python -m pip check

COPY server.py /service/server.py

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
