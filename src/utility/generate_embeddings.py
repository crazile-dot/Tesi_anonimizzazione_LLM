
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_BASELINE = DATA_DIR / "dataset_680_baseline.jsonl"
INPUT_M4 = DATA_DIR / "openai_results_680_m4_v2.jsonl"
INPUT_M5 = DATA_DIR / "openai_results_680_m5_v2.jsonl"

OUTPUT_FILE = DATA_DIR / "embeddings_680_all_v1.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE_TOKENS = 224
CHUNK_OVERLAP_TOKENS = 32
MAX_MODEL_TOKENS = 224
MODEL_MAX_SEQ_LENGTH = 256


# ============================================================
# LETTURA JSONL
# ============================================================

def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(
            f"File di input non trovato: {path}"
        )

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON non valido in {path.name}, "
                    f"riga {line_number}: {exc}"
                ) from exc

    return records


def index_by_document_id(records, label):
    indexed = {}

    for record in records:
        document_id = record.get("document_id")

        if not document_id:
            raise ValueError(
                f"document_id mancante nel dataset {label}"
            )

        if document_id in indexed:
            raise ValueError(
                f"document_id duplicato in {label}: {document_id}"
            )

        indexed[document_id] = record

    return indexed


# ============================================================
# CHUNKING UNIFORME
# ============================================================

def create_chunks(text, tokenizer):
    """
    Chunking token-aware uniforme per M0-M5.

    Dimensione massima: 224 token, esclusi i token speciali.
    Overlap nominale: 32 token.
    La lunghezza viene verificata dopo il decode.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    if not token_ids:
        return []

    chunks = []
    start = 0

    while start < len(token_ids):
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

            actual_token_ids = tokenizer.encode(
                chunk,
                add_special_tokens=False
            )

            if len(actual_token_ids) <= MAX_MODEL_TOKENS:
                break

            reduction = max(
                1,
                len(actual_token_ids) - MAX_MODEL_TOKENS
            )

            end -= reduction

        if end <= start:
            raise RuntimeError(
                "Impossibile creare un chunk valido."
            )

        chunks.append(chunk)

        if end == len(token_ids):
            break

        start = end - CHUNK_OVERLAP_TOKENS

    return chunks


# ============================================================
# EMBEDDING DOCUMENT-LEVEL
# ============================================================

def document_embedding(text, model):
    chunks = create_chunks(
        text,
        model.tokenizer
    )

    if not chunks:
        return {
            "num_chunks": 0,
            "embedding": None,
            "status": "empty_text"
        }

    chunk_embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False
    )

    document_vector = np.mean(
        chunk_embeddings,
        axis=0
    )

    norm = np.linalg.norm(document_vector)

    if norm > 0:
        document_vector = document_vector / norm

    return {
        "num_chunks": len(chunks),
        "embedding": document_vector.tolist(),
        "status": "success"
    }


# ============================================================
# ESTRAZIONE TESTI PER METODO
# ============================================================

def get_method_texts(baseline, record_m4, record_m5):
    return {
        "M0_original": baseline.get("M0_originale"),
        "M1_redaction": baseline.get("M1_redacted"),
        "M2_pseudonymization": baseline.get("M2_persona"),
        "M3_hmac": baseline.get("M3_hmac"),
        "M4_llm_substitution": (
            record_m4.get("text_anonymized")
            if record_m4 is not None
            else None
        ),
        "M5_llm_anonymization": (
            record_m5.get("text_anonymized")
            if record_m5 is not None
            else None
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Lettura dei tre dataset...")

    baseline_records = load_jsonl(INPUT_BASELINE)
    m4_records = load_jsonl(INPUT_M4)
    m5_records = load_jsonl(INPUT_M5)

    baseline = index_by_document_id(
        baseline_records, "M0-M3"
    )
    m4 = index_by_document_id(m4_records, "M4")
    m5 = index_by_document_id(m5_records, "M5")

    baseline_ids = set(baseline)
    m4_ids = set(m4)
    m5_ids = set(m5)

    print(f"Documenti M0-M3: {len(baseline)}")
    print(f"Documenti M4:    {len(m4)}")
    print(f"Documenti M5:    {len(m5)}")

    if baseline_ids != m4_ids or baseline_ids != m5_ids:
        raise ValueError(
            "Gli insiemi di document_id non coincidono. "
            "Verificare gli input prima di procedere."
        )

    document_ids = list(baseline.keys())

    # Controllo preventivo degli stati M4/M5.
    # Non interrompiamo il lavoro per un singolo record fallito:
    # lo conserviamo nell'output con embedding non disponibile.
    for label, records in [("M4", m4), ("M5", m5)]:
        statuses = {}

        for record in records.values():
            status = record.get("status", "<missing>")
            statuses[status] = statuses.get(status, 0) + 1

        print(f"Stati {label}: {statuses}")

    print(f"\nCaricamento modello: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MODEL_MAX_SEQ_LENGTH

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nGenerazione embedding per {len(document_ids)} documenti...")

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for i, document_id in enumerate(document_ids, start=1):
            texts = get_method_texts(
                baseline[document_id],
                m4.get(document_id),
                m5.get(document_id)
            )

            embeddings = {}

            for method, text in texts.items():
                # Per M4/M5 non utilizziamo output non riusciti.
                if method.startswith("M4_") or method.startswith("M5_"):
                    source = m4 if method.startswith("M4_") else m5
                    source_record = source.get(document_id)

                    if (
                        source_record is None
                        or source_record.get("status") != "success"
                    ):
                        embeddings[method] = {
                            "num_chunks": 0,
                            "embedding": None,
                            "status": "source_not_successful"
                        }
                        continue

                embeddings[method] = document_embedding(
                    text, model
                )

            result = {
                "document_id": document_id,
                "model": MODEL_NAME,
                "chunk_size_tokens": CHUNK_SIZE_TOKENS,
                "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
                "max_model_tokens": MAX_MODEL_TOKENS,
                "aggregation": "mean_then_l2_normalization",
                "embeddings": embeddings
            }

            out.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

            if i % 25 == 0 or i == len(document_ids):
                print(f"Elaborati: {i}/{len(document_ids)}")

    print("\nGenerazione completata.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
