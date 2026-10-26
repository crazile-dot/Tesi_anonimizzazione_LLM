
import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_M0 = BASE_DIR / "data" / "processed" / "embeddings_100.jsonl"
INPUT_M5 = BASE_DIR / "data" / "processed" / "embeddings_m5_100.jsonl"

OUTPUT_PER_DOCUMENT = BASE_DIR / "data" / "processed" / "similarity_m5_100.jsonl"
OUTPUT_SUMMARY = BASE_DIR / "data" / "processed" / "similarity_m5_summary.json"

M0_KEY = "M0_original"
M5_KEY = "M5_llm_anonymization"


def load_jsonl(path):
    records = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            document_id = record["document_id"]

            if document_id in records:
                raise ValueError(f"ID duplicato in {path.name}: {document_id}")

            records[document_id] = record

    return records


def get_embedding(record, key):
    try:
        return np.asarray(
            record["embeddings"][key]["embedding"],
            dtype=float
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Embedding '{key}' mancante per "
            f"{record.get('document_id', 'ID sconosciuto')}"
        ) from exc


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Trovato un embedding con norma zero.")

    if a.shape != b.shape:
        raise ValueError(
            f"Dimensioni incompatibili: {a.shape} e {b.shape}"
        )

    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    for path in (INPUT_M0, INPUT_M5):
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")

    m0_records = load_jsonl(INPUT_M0)
    m5_records = load_jsonl(INPUT_M5)

    ids_m0 = set(m0_records)
    ids_m5 = set(m5_records)

    if ids_m0 != ids_m5:
        only_m0 = sorted(ids_m0 - ids_m5)
        only_m5 = sorted(ids_m5 - ids_m0)
        raise ValueError(
            "Gli ID dei due file non coincidono.\n"
            f"Solo M0: {only_m0[:10]}\n"
            f"Solo M5: {only_m5[:10]}"
        )

    if len(ids_m0) != 100:
        raise ValueError(
            f"Attesi 100 documenti, trovati {len(ids_m0)}."
        )

    results = []

    for document_id in sorted(ids_m0):
        embedding_m0 = get_embedding(m0_records[document_id], M0_KEY)
        embedding_m5 = get_embedding(m5_records[document_id], M5_KEY)

        similarity = cosine_similarity(embedding_m0, embedding_m5)

        results.append({
            "document_id": document_id,
            "M0_vs_M5": similarity
        })

    values = np.array(
        [record["M0_vs_M5"] for record in results],
        dtype=float
    )

    with open(OUTPUT_PER_DOCUMENT, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "comparison": "M0_vs_M5",
        "embedding_model": "all-MiniLM-L6-v2",
        "documents": len(results),
        "mean_cosine_similarity": float(np.mean(values)),
        "median_cosine_similarity": float(np.median(values)),
        "std_cosine_similarity": float(np.std(values)),
        "min_cosine_similarity": float(np.min(values)),
        "max_cosine_similarity": float(np.max(values))
    }

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== SIMILARITY M0–M5 ===")
    print(f"Documenti confrontati: {len(results)}")
    print(f"Media:   {summary['mean_cosine_similarity']:.4f}")
    print(f"Mediana: {summary['median_cosine_similarity']:.4f}")
    print(f"Dev. std.: {summary['std_cosine_similarity']:.4f}")
    print(f"Minimo:  {summary['min_cosine_similarity']:.4f}")
    print(f"Massimo: {summary['max_cosine_similarity']:.4f}")
    print(f"\nRisultati per documento: {OUTPUT_PER_DOCUMENT}")
    print(f"Riepilogo: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
