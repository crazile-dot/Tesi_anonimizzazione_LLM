
import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_MAIN = DATA_DIR / "embeddings_100.jsonl"
INPUT_M4 = DATA_DIR / "embeddings_m4_100.jsonl"
INPUT_M5 = DATA_DIR / "embeddings_m5_100.jsonl"

OUTPUT_PER_DOCUMENT = DATA_DIR / "retrieval_stability_m5_per_document.csv"
OUTPUT_SUMMARY = DATA_DIR / "retrieval_stability_m5_summary.csv"

K = 5

METHODS = {
    "M0": ("main", "M0_original"),
    "M1": ("main", "M1_redaction"),
    "M2": ("main", "M2_pseudonymization"),
    "M3": ("main", "M3_hmac"),
    "M4": ("m4", "M4_llm_substitution"),
    "M5": ("m5", "M5_llm_anonymization"),
}


def load_jsonl(path):
    records = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
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


def get_embedding(record, key):
    try:
        vector = np.asarray(
            record["embeddings"][key]["embedding"],
            dtype=float
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Embedding '{key}' mancante per "
            f"{record.get('document_id', 'ID sconosciuto')}"
        ) from exc

    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"Embedding non valido per {record.get('document_id')}"
        )

    if np.linalg.norm(vector) == 0:
        raise ValueError(
            f"Embedding con norma zero per {record.get('document_id')}"
        )

    return vector


def ranking(query_vector, corpus, excluded_id, k):
    scored = []

    for document_id, vector in corpus.items():
        if document_id == excluded_id:
            continue

        if query_vector.shape != vector.shape:
            raise ValueError(
                f"Dimensioni incompatibili per {document_id}: "
                f"{query_vector.shape} vs {vector.shape}"
            )

        similarity = float(
            np.dot(query_vector, vector)
            / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
        )

        scored.append((document_id, similarity))

    # Ordinamento deterministico anche in caso di parità
    scored.sort(key=lambda item: (-item[1], item[0]))

    return [document_id for document_id, _ in scored[:k]]


def main():
    for path in (INPUT_MAIN, INPUT_M4, INPUT_M5):
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")

    main_records = load_jsonl(INPUT_MAIN)
    m4_records = load_jsonl(INPUT_M4)
    m5_records = load_jsonl(INPUT_M5)

    id_sets = {
        "main": set(main_records),
        "m4": set(m4_records),
        "m5": set(m5_records),
    }

    if not (
        id_sets["main"] == id_sets["m4"] == id_sets["m5"]
    ):
        raise ValueError(
            "Gli ID dei file embedding non coincidono. "
            f"Numero documenti: main={len(id_sets['main'])}, "
            f"M4={len(id_sets['m4'])}, M5={len(id_sets['m5'])}"
        )

    document_ids = sorted(id_sets["main"])

    if len(document_ids) != 100:
        raise ValueError(
            f"Attesi 100 documenti, trovati {len(document_ids)}."
        )

    sources = {
        "main": main_records,
        "m4": m4_records,
        "m5": m5_records,
    }

    embeddings = {}

    for method, (source_name, embedding_key) in METHODS.items():
        embeddings[method] = {
            document_id: get_embedding(
                sources[source_name][document_id],
                embedding_key
            )
            for document_id in document_ids
        }

    # Le query sono sempre gli embedding originali M0.
    # Cambia il corpus sul quale vengono cercati i vicini.
    per_document = []

    for query_id in document_ids:
        query_vector = embeddings["M0"][query_id]

        baseline_top_k = ranking(
            query_vector,
            embeddings["M0"],
            query_id,
            K
        )
        baseline_set = set(baseline_top_k)

        for method in METHODS:
            method_top_k = ranking(
                query_vector,
                embeddings[method],
                query_id,
                K
            )
            method_set = set(method_top_k)

            intersection = len(baseline_set & method_set)
            union = len(baseline_set | method_set)

            recall = intersection / K
            jaccard = intersection / union if union else 0.0

            per_document.append({
                "document_id": query_id,
                "method": method,
                "k": K,
                "recall_at_k": recall,
                "jaccard_at_k": jaccard,
            })

    # Scrittura dei risultati per documento
    import csv

    with open(
        OUTPUT_PER_DOCUMENT, "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "document_id",
                "method",
                "k",
                "recall_at_k",
                "jaccard_at_k",
            ],
        )
        writer.writeheader()
        writer.writerows(per_document)

    # Riepilogo per metodo
    summary = []

    for method in METHODS:
        rows = [
            row for row in per_document
            if row["method"] == method
        ]

        summary.append({
            "method": method,
            "documents": len(rows),
            "k": K,
            "mean_recall_at_k": float(
                np.mean([row["recall_at_k"] for row in rows])
            ),
            "mean_jaccard_at_k": float(
                np.mean([row["jaccard_at_k"] for row in rows])
            ),
        })

    with open(
        OUTPUT_SUMMARY, "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "documents",
                "k",
                "mean_recall_at_k",
                "mean_jaccard_at_k",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\n=== STABILITÀ DEL RETRIEVAL — CAMPIONE 100 ===")
    print("Query: embedding originale M0")
    print(f"Vicini considerati: Top-{K}\n")

    for row in summary:
        print(
            f"{row['method']}: "
            f"Recall@{K}={row['mean_recall_at_k']:.4f} | "
            f"Jaccard@{K}={row['mean_jaccard_at_k']:.4f}"
        )

    print(f"\nRisultati per documento: {OUTPUT_PER_DOCUMENT}")
    print(f"Riepilogo: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
