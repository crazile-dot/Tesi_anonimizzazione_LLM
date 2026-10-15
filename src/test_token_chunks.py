import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_FILE = BASE_DIR / "data" / "processed" / "dataset_100_baseline.jsonl"

TARGET = "baughman-e_all_documents_13"

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE_TOKENS = 224
CHUNK_OVERLAP_TOKENS = 32


def create_chunks(text, tokenizer):
    if not text.strip():
        return []

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []
    start = 0

    while start < len(token_ids):
        end = min(start + CHUNK_SIZE_TOKENS, len(token_ids))

        chunk_token_ids = token_ids[start:end]

        chunk = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=True
        )

        chunks.append(chunk)

        if end == len(token_ids):
            break

        start = end - CHUNK_OVERLAP_TOKENS

    return chunks


print("Caricamento modello...")
model = SentenceTransformer(MODEL_NAME)

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        document = json.loads(line)

        if document["document_id"] != TARGET:
            continue

        text = document["M0_original"]

        chunks = create_chunks(text, model.tokenizer)

        print()
        print("========================================")
        print("TEST TOKEN CHUNKING")
        print("========================================")
        print(f"Documento: {TARGET}")
        print(f"Token totali: {len(model.tokenizer.encode(text, add_special_tokens=False))}")
        print(f"Numero chunk: {len(chunks)}")
        print()

        for i, chunk in enumerate(chunks, start=1):
            token_count = len(
                model.tokenizer.encode(
                    chunk,
                    add_special_tokens=False
                )
            )

            print(f"Chunk {i}: {token_count} token")
            print(f"  Caratteri: {len(chunk)}")
            print(f"  Inizio: {chunk[:100]}...")
            print()