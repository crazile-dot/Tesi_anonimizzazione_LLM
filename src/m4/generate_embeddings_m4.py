import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "openai_results_100.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "embeddings_m4_100.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"

# Manteniamo la stessa configurazione usata per M0-M3.
CHUNK_SIZE_TOKENS = 224
CHUNK_OVERLAP_TOKENS = 32

# Limite massimo effettivo dei chunk passati al modello.
# Usiamo 224 per lasciare margine rispetto al limite di 256
# token del modello.
MAX_MODEL_TOKENS = 224


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(text, tokenizer):
    """
    Divide il testo in chunk token-aware.

    La dimensione nominale è di 224 token con overlap di 32 token.
    Dopo il decode, il chunk viene nuovamente tokenizzato per
    verificare la sua lunghezza effettiva prima di passarlo
    a SentenceTransformer.

    Questo evita che un chunk che nasce da 224 token superi
    accidentalmente il limite del modello dopo il decode/re-tokenize.
    """

    if not text or not text.strip():
        return []

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []
    start = 0

    while start < len(token_ids):

        # Tentativo iniziale con la dimensione nominale.
        end = min(
            start + CHUNK_SIZE_TOKENS,
            len(token_ids)
        )

        while end > start:

            chunk_token_ids = token_ids[start:end]

            chunk = tokenizer.decode(
                chunk_token_ids,
                skip_special_tokens=True
            )

            # Controlliamo quanti token produce realmente
            # il tokenizer quando il testo viene nuovamente
            # passato al modello.
            actual_token_ids = tokenizer.encode(
                chunk,
                add_special_tokens=False
            )

            actual_length = len(actual_token_ids)

            if actual_length <= MAX_MODEL_TOKENS:
                break

            # Se il chunk è ancora troppo lungo, lo riduciamo.
            # La riduzione progressiva evita di superare il limite.
            reduction = max(
                1,
                actual_length - MAX_MODEL_TOKENS
            )

            end -= reduction

        if end <= start:
            raise RuntimeError(
                "Impossibile creare un chunk entro il limite "
                f"di {MAX_MODEL_TOKENS} token."
            )

        chunks.append(chunk)

        if end == len(token_ids):
            break

        # Overlap nominale di 32 token.
        start = end - CHUNK_OVERLAP_TOKENS

        # Sicurezza contro eventuali loop.
        if start < 0:
            start = 0

    return chunks


# ============================================================
# EMBEDDING DEL DOCUMENTO
# ============================================================

def document_embedding(text, model):
    """
    Genera un embedding document-level facendo la media
    degli embedding dei chunk e successiva normalizzazione L2.
    """

    chunks = create_chunks(
        text,
        model.tokenizer
    )

    if not chunks:
        return [], None

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # Media degli embedding dei chunk
    document_vector = np.mean(
        embeddings,
        axis=0
    )

    # Normalizzazione L2
    norm = np.linalg.norm(
        document_vector
    )

    if norm > 0:
        document_vector = document_vector / norm

    return chunks, document_vector.tolist()


# ============================================================
# MAIN
# ============================================================

def main():

    print("Caricamento modello...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    # Manteniamo il limite del modello.
    model.max_seq_length = 256

    print(
        f"Lettura dataset: {INPUT_FILE}"
    )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        documents = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    print(
        f"Documenti da elaborare: {len(documents)}"
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as out:

        for i, document in enumerate(
            documents,
            start=1
        ):

            document_id = document["document_id"]

            print(
                f"[{i}/{len(documents)}] {document_id}"
            )

            text_m4 = document["text_anonymized"]

            chunks, embedding = document_embedding(
                text_m4,
                model
            )

            result = {
                "document_id": document_id,
                "model": MODEL_NAME,
                "chunk_size_tokens": CHUNK_SIZE_TOKENS,
                "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
                "embeddings": {
                    "M4_llm_substitution": {
                        "num_chunks": len(chunks),
                        "embedding": embedding
                    }
                }
            }

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

    print()
    print("Embedding M4 completati.")
    print(
        f"Output salvato in: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()