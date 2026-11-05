from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

torch.set_num_threads(2)
model = SentenceTransformer(MODEL_ID, device="cpu")
lock = Lock()
app = FastAPI(title="ConfLlm temporary embeddings")


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str = "float"


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": "cpu"}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_ID, "object": "model", "owned_by": "local"}
        ],
    }


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest):
    if request.model != MODEL_ID:
        raise HTTPException(400, "Unknown embedding model")

    if request.encoding_format != "float":
        raise HTTPException(400, "Only float encoding is supported")

    texts = (
        [request.input]
        if isinstance(request.input, str)
        else request.input
    )

    if not 1 <= len(texts) <= 32:
        raise HTTPException(400, "Expected 1 to 32 texts")

    if any(not text.strip() or len(text) > 20000 for text in texts):
        raise HTTPException(400, "Empty or oversized input")

    with lock:
        vectors = model.encode(
            texts,
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    return {
        "object": "list",
        "model": MODEL_ID,
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": vector,
            }
            for index, vector in enumerate(vectors)
        ],
    }


# ConfLlm: token-aware chunks v1
class ChunkRequest(BaseModel):
    text: str


@app.post("/v1/chunks")
def chunks(request: ChunkRequest):
    text = request.text
    if not text.strip():
        return {"chunks": [], "max_seq_length": model.max_seq_length}

    with lock:
        tokenizer = model.tokenizer
        if not tokenizer.is_fast:
            raise HTTPException(500, "Fast tokenizer is required")

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        limit = int(model.max_seq_length)
        special = tokenizer.num_special_tokens_to_add(pair=False)
        budget = min(96, limit - special)

        if budget <= 24:
            raise HTTPException(500, "Unexpected model sequence limit")

        result = []
        start = 0

        while start < len(offsets):
            stop = min(start + budget, len(offsets))

            while stop > start:
                left = offsets[start][0]
                right = offsets[stop - 1][1]
                fragment = text[left:right]

                token_count = len(tokenizer(
                    fragment,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"])

                if token_count <= limit:
                    break
                stop -= 1

            if stop <= start:
                raise HTTPException(500, "Cannot create a safe chunk")

            if fragment.strip():
                result.append({
                    "text": fragment,
                    "token_count": token_count,
                    "char_start": left,
                    "char_end": right,
                })

            if stop == len(offsets):
                break

            start = max(start + 1, stop - 24)

    return {"chunks": result, "max_seq_length": limit}
