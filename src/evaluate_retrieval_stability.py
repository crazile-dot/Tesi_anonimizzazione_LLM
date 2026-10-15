
import csv
import json
import math
from pathlib import Path
from statistics import mean

BASE_DIR = Path("data/processed")
INPUT_MAIN = BASE_DIR / "embeddings_100.jsonl"
INPUT_M4 = BASE_DIR / "embeddings_m4_100.jsonl"

OUTPUT_PER_DOCUMENT = BASE_DIR / "retrieval_stability_per_document.csv"
OUTPUT_SUMMARY = BASE_DIR / "retrieval_stability_summary.csv"

K = 5

METHODS = {
    "M0": ("main", "M0_original"),
    "M1": ("main", "M1_redaction"),
    "M2": ("main", "M2_pseudonymization"),
    "M3": ("main", "M3_hmac"),
    "M4": ("m4", "M4_llm_substitution"),
}


def load_jsonl(path):
    records = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            doc_id = record["document_id"]

            if doc_id in records:
                raise ValueError(f"ID duplicato in {path}: {doc_id}")

            records[doc_id] = record

    return records


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Embedding con norma zero.")

    return dot / (norm_a * norm_b)


def get_embedding(record, method):
    vector = record["embeddings"][method]["embedding"]

    if not isinstance(vector, list) or not vector:
        raise ValueError(
            f"Embedding non valido: {record['document_id']} - {method}"
        )

    return vector


def ranking(query_vector, corpus, method, excluded_id):
    scores = []

    for doc_id, record in corpus.items():
        if doc_id == excluded_id:
            continue

        vector = get_embedding(record, method)

        if len(vector) != len(query_vector):
            raise ValueError(
                f"Dimensioni incompatibili per {doc_id}, metodo {method}"
            )

        score = cosine_similarity(query_vector, vector)
        scores.append((doc_id, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in scores]


def main():
    if not INPUT_MAIN.exists() or not INPUT_M4.exists():
        raise FileNotFoundError(
            "Non trovo entrambi i file degli embedding."
        )

    main_records = load_jsonl(INPUT_MAIN)
    m4_records = load_jsonl(INPUT_M4)

    if set(main_records) != set(m4_records):
        raise ValueError(
            "Gli ID dei documenti non coincidono tra i due file."
        )

    doc_ids = sorted(main_records)

    if len(doc_ids) <= K:
        raise ValueError(f"Servono più di {K} documenti.")

    for doc_id in doc_ids:
        for method in [
            "M0_original",
            "M1_redaction",
            "M2_pseudonymization",
            "M3_hmac",
        ]:
            get_embedding(main_records[doc_id], method)

        get_embedding(m4_records[doc_id], "M4_llm_substitution")

    results = []

    for query_id in doc_ids:
        # La query è sempre l'embedding del documento originale.
        query_vector = get_embedding(
            main_records[query_id], "M0_original"
        )

        baseline = ranking(
            query_vector, main_records, "M0_original", query_id
        )[:K]

        for label, (source, embedding_key) in METHODS.items():
            corpus = m4_records if source == "m4" else main_records

            retrieved = ranking(
                query_vector, corpus, embedding_key, query_id
            )[:K]

            intersection = len(set(baseline) & set(retrieved))
            recall_at_k = intersection / len(baseline)

            union = len(set(baseline) | set(retrieved))
            jaccard_at_k = intersection / union if union else 0.0

            results.append({
                "query_document": query_id,
                "method": label,
                "k": K,
                "recall_at_k": recall_at_k,
                "jaccard_at_k": jaccard_at_k,
            })

    fields = [
        "query_document",
        "method",
        "k",
        "recall_at_k",
        "jaccard_at_k",
    ]

    with OUTPUT_PER_DOCUMENT.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    summary = []

    for label in METHODS:
        subset = [r for r in results if r["method"] == label]

        summary.append({
            "method": label,
            "documents": len(subset),
            "k": K,
            "mean_recall_at_k": mean(
                r["recall_at_k"] for r in subset
            ),
            "mean_jaccard_at_k": mean(
                r["jaccard_at_k"] for r in subset
            ),
        })

    with OUTPUT_SUMMARY.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "method",
                "documents",
                "k",
                "mean_recall_at_k",
                "mean_jaccard_at_k",
            ]
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"Documenti valutati: {len(doc_ids)}")
    print(f"Vicini di riferimento: top-{K} su M0")
    print("\nRisultati medi:")

    for row in summary:
        print(
            f"{row['method']}: "
            f"Recall@{K}={row['mean_recall_at_k']:.4f}, "
            f"Jaccard@{K}={row['mean_jaccard_at_k']:.4f}"
        )

    print(f"\nReport per documento: {OUTPUT_PER_DOCUMENT}")
    print(f"Riepilogo: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
