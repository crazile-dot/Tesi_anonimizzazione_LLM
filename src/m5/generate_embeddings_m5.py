
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "openai_results_100_m5.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "embeddings_m5_100.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE_TOKENS = 224
CHUNK_OVERLAP_TOKENS = 32
MAX_MODEL_TOKENS = 224

EMBEDDING_KEY = "M5_llm_anonymization"


# ============================================================
# LETTURA DEI DOCUMENTI
# ============================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON non valido alla riga {line_number}: {exc}"
                ) from exc

    return records


# ============================================================
# SUDDIVISIONE IN CHUNK
# ============================================================

def create_chunks(text, tokenizer):
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    if not token_ids:
        return [""]

    chunks = []
    step = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS

    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start:start + CHUNK_SIZE_TOKENS]

        if not chunk_ids:
            break

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        chunks.append(chunk_text)

        if start + CHUNK_SIZE_TOKENS >= len(token_ids):
            break

    return chunks


# ============================================================
# EMBEDDING DEL DOCUMENTO
# ============================================================

def document_embedding(text, model):
    chunks = create_chunks(text, model.tokenizer)

    chunk_embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False
    )

    # Media degli embedding dei chunk
    document_vector = np.mean(chunk_embeddings, axis=0)

    # Normalizzazione L2 del vettore finale
    norm = np.linalg.norm(document_vector)

    if norm > 0:
        document_vector = document_vector / norm

    return document_vector, len(chunks)


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File di input non trovato: {INPUT_FILE}"
        )

    documents = load_jsonl(INPUT_FILE)

    print(f"Documenti letti: {len(documents)}")

    if len(documents) != 100:
        raise ValueError(
            f"Attesi 100 documenti M5, trovati {len(documents)}."
        )

    document_ids = [d["document_id"] for d in documents]

    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Sono presenti document_id duplicati.")

    for document in documents:
        if document.get("status") != "success":
            raise ValueError(
                f"Documento non elaborato correttamente: "
                f"{document.get('document_id')} "
                f"(status={document.get('status')})"
            )

        if not document.get("text_anonymized", "").strip():
            raise ValueError(
                f"Testo M5 vuoto: {document.get('document_id')}"
            )

    print(f"Caricamento modello: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 256

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    processed = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for document in documents:
            document_id = document["document_id"]
            text = document["text_anonymized"]

            vector, num_chunks = document_embedding(text, model)

            result = {
                "document_id": document_id,
                "model": MODEL_NAME,
                "chunk_size_tokens": CHUNK_SIZE_TOKENS,
                "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
                "embeddings": {
                    EMBEDDING_KEY: {
                        "num_chunks": num_chunks,
                        "embedding": vector.tolist()
                    }
                }
            }

            out.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

            processed += 1

            if processed % 10 == 0 or processed == len(documents):
                print(f"Elaborati: {processed}/{len(documents)}")

    print("\nGenerazione embedding M5 completata.")
    print(f"Documenti elaborati: {processed}")
    print(f"File salvato: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
