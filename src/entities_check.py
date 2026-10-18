
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

BASELINE_FILE = DATA_DIR / "dataset_680_baseline.jsonl"
M4_FILE = DATA_DIR / "openai_results_680_m4_v2.jsonl"
M5_FILE = DATA_DIR / "openai_results_680_m5_v2.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 224
CHUNK_OVERLAP = 32

METHODS = {
    "M0_original": ("baseline", "M0_originale"),
    "M1_redaction": ("baseline", "M1_redacted"),
    "M2_pseudonymization": ("baseline", "M2_persona"),
    "M3_hmac": ("baseline", "M3_hmac"),
    "M4_llm_substitution": ("m4", "text_anonymized"),
    "M5_llm_anonymization": ("m5", "text_anonymized"),
}


def load_jsonl(path):
    records = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            document_id = record["document_id"]
            if document_id in records:
                raise ValueError(
                    f"ID duplicato in {path.name}: {document_id}"
                )
            records[document_id] = record
    return records


def create_chunks(text, tokenizer):
    if not isinstance(text, str) or not text.strip():
        return []

    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))

        while end > start:
            chunk = tokenizer.decode(
                tokens[start:end],
                skip_special_tokens=True
            )
            chunk_tokens = tokenizer.encode(
                chunk,
                add_special_tokens=False
            )

            if len(chunk_tokens) <= CHUNK_SIZE:
                break

            end -= 1

        if end <= start:
            raise ValueError("Impossibile creare un chunk valido.")

        chunks.append(chunk)

        if end >= len(tokens):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main():
    print("Lettura dei dataset...")
    baseline = load_jsonl(BASELINE_FILE)
    m4 = load_jsonl(M4_FILE)
    m5 = load_jsonl(M5_FILE)

    print("Documenti baseline:", len(baseline))
    print("Documenti M4:", len(m4))
    print("Documenti M5:", len(m5))

    if set(baseline) != set(m4) or set(baseline) != set(m5):
        raise ValueError("Gli ID dei tre dataset non coincidono.")

    print("\nCaricamento modello e tokenizer...")
    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer

    results = {
        method: {
            "documents": 0,
            "chunks": 0,
            "max_tokens": 0,
            "over_limit": 0,
            "skipped": 0,
        }
        for method in METHODS
    }
    errors = []

    print("\nControllo dei chunk...")
    for document_id in baseline:
        sources = {
            "baseline": baseline[document_id],
            "m4": m4[document_id],
            "m5": m5[document_id],
        }

        for method, (source_name, field) in METHODS.items():
            record = sources[source_name]
            stats = results[method]

            if source_name in ("m4", "m5") and record.get("status") != "success":
                stats["skipped"] += 1
                continue

            text = record.get(field)

            if not isinstance(text, str) or not text.strip():
                stats["skipped"] += 1
                continue

            chunks = create_chunks(text, tokenizer)
            stats["documents"] += 1

            for index, chunk in enumerate(chunks):
                length = len(
                    tokenizer.encode(
                        chunk,
                        add_special_tokens=False
                    )
                )
                stats["chunks"] += 1
                stats["max_tokens"] = max(
                    stats["max_tokens"], length
                )

                if length > CHUNK_SIZE:
                    stats["over_limit"] += 1
                    errors.append(
                        (method, document_id, index, length)
                    )

    print("\n--- RISULTATI PER STRATEGIA ---")
    for method, stats in results.items():
        print(f"\n{method}")
        print("  Documenti controllati:", stats["documents"])
        print("  Documenti saltati:", stats["skipped"])
        print("  Chunk controllati:", stats["chunks"])
        print("  Lunghezza massima:", stats["max_tokens"])
        print("  Chunk oltre il limite:", stats["over_limit"])

    print("\nAnomalie complessive:", len(errors))
    if errors:
        print("Prime anomalie (strategia, ID, chunk, token):")
        for error in errors[:20]:
            print(" ", error)
    else:
        print("OK: nessun chunk supera il limite di 224 token.")


if __name__ == "__main__":
    main()
