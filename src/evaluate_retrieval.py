
import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

EMBEDDINGS_FILE = DATA_DIR / "embeddings_100.jsonl"
EMBEDDINGS_M4_FILE = DATA_DIR / "embeddings_m4_100.jsonl"
DATASET_FILE = DATA_DIR / "dataset_100_baseline.jsonl"

OUTPUT_FILE = DATA_DIR / "ranking_stability_100.jsonl"
SUMMARY_FILE = DATA_DIR / "ranking_stability_summary.json"

METHODS = [
    "M1_redaction",
    "M2_pseudonymization",
    "M3_hmac",
    "M4_llm_substitution",
]

KS = [1, 3, 5, 10]
MRR_K = 5


# ============================================================
# CARICAMENTO JSONL
# ============================================================

def load_jsonl(filepath):
    records = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            document_id = record["document_id"]

            if document_id in records:
                raise ValueError(
                    f"ID duplicato in {filepath.name}: "
                    f"{document_id}"
                )

            records[document_id] = record

    return records


# ============================================================
# EMBEDDING E RANKING
# ============================================================

def cosine_similarity_matrix(embeddings):
    matrix = np.asarray(embeddings, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("Gli embedding devono formare una matrice 2D.")

    if not np.isfinite(matrix).all():
        raise ValueError("Sono presenti valori non finiti negli embedding.")

    norms = np.linalg.norm(matrix, axis=1)

    if np.any(norms == 0):
        raise ValueError("Sono presenti embedding con norma zero.")

    # Normalizzazione esplicita: non dipendiamo dal fatto
    # che i vettori siano già normalizzati nei file.
    matrix = matrix / norms[:, None]

    return matrix @ matrix.T


def get_ranking(similarity_matrix, query_index):
    similarities = similarity_matrix[query_index].copy()

    # Esclude il documento usato come query.
    similarities[query_index] = -np.inf

    return np.argsort(-similarities)


# ============================================================
# METRICHE DI STABILITA' DEL RANKING
# ============================================================

def ranking_overlap(reference_ranking, transformed_ranking, k):
    """
    Frazione dei top-k di M0 mantenuta nei top-k trasformati.
    È equivalente alla Recall@k usata nello script precedente.
    """
    reference_top_k = set(reference_ranking[:k])
    transformed_top_k = set(transformed_ranking[:k])

    if not reference_top_k:
        return 0.0

    return (
        len(reference_top_k & transformed_top_k)
        / len(reference_top_k)
    )


def reciprocal_rank_reference(reference_ranking, transformed_ranking, k):
    """
    Reciproco della posizione del primo documento dei top-k di M0
    presente nei top-k trasformati. Vale 0 se non c'è sovrapposizione.
    """
    reference_top_k = set(reference_ranking[:k])

    for rank, document_index in enumerate(
        transformed_ranking[:k], start=1
    ):
        if document_index in reference_top_k:
            return 1.0 / rank

    return 0.0


# ============================================================
# MAIN
# ============================================================

def main():
    print("Caricamento dei dati...")

    baseline_records = load_jsonl(EMBEDDINGS_FILE)
    m4_records = load_jsonl(EMBEDDINGS_M4_FILE)
    dataset_records = load_jsonl(DATASET_FILE)

    # Stesso campione per tutti i metodi:
    # documenti con PERSON annotati e embedding M0/M4 disponibili.
    document_ids = []

    for document_id, dataset_record in dataset_records.items():
        if not dataset_record.get("entities", []):
            continue

        if document_id not in baseline_records:
            continue

        if document_id not in m4_records:
            continue

        document_ids.append(document_id)

    if len(document_ids) < 2:
        raise ValueError(
            "Documenti insufficienti per calcolare i ranking."
        )

    print(f"Documenti nel dataset: {len(dataset_records)}")
    print(f"Documenti con PERSON e embedding M0/M4: {len(document_ids)}")
    print()

    # --------------------------------------------------------
    # Ranking di riferimento M0
    # --------------------------------------------------------

    m0_embeddings = [
        baseline_records[doc_id]["embeddings"]
        ["M0_original"]["embedding"]
        for doc_id in document_ids
    ]

    m0_similarity = cosine_similarity_matrix(m0_embeddings)

    m0_rankings = [
        get_ranking(m0_similarity, i)
        for i in range(len(document_ids))
    ]

    # --------------------------------------------------------
    # Ranking e metriche per ogni metodo
    # --------------------------------------------------------

    all_results = []
    summary = {}

    for method in METHODS:
        print(f"Calcolo: {method}")

        method_embeddings = []

        for document_id in document_ids:
            if method == "M4_llm_substitution":
                record = m4_records[document_id]
            else:
                record = baseline_records[document_id]

            if method not in record.get("embeddings", {}):
                raise KeyError(
                    f"Embedding {method} mancante per {document_id}"
                )

            method_embeddings.append(
                record["embeddings"][method]["embedding"]
            )

        method_similarity = cosine_similarity_matrix(
            method_embeddings
        )

        overlaps = {k: [] for k in KS}
        mrr_values = []
        method_results = []

        for i, document_id in enumerate(document_ids):
            transformed_ranking = get_ranking(method_similarity, i)
            reference_ranking = m0_rankings[i]

            result = {
                "document_id": document_id,
                "method": method,
            }

            for k in KS:
                value = ranking_overlap(
                    reference_ranking,
                    transformed_ranking,
                    k,
                )

                result[f"ranking_overlap@{k}"] = value
                overlaps[k].append(value)

            mrr = reciprocal_rank_reference(
                reference_ranking,
                transformed_ranking,
                MRR_K,
            )

            result[f"mrr_reference@{MRR_K}"] = mrr
            mrr_values.append(mrr)
            method_results.append(result)

        all_results.extend(method_results)

        summary[method] = {
            "n_documents": len(method_results),
            **{
                f"ranking_overlap@{k}": float(np.mean(overlaps[k]))
                for k in KS
            },
            f"mrr_reference@{MRR_K}": float(np.mean(mrr_values)),
        }

    # --------------------------------------------------------
    # Salvataggio
    # --------------------------------------------------------

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n=== STABILITA' DEL RANKING SEMANTICO ===")

    for method, values in summary.items():
        print(f"\n{method}")
        print(f"  Documenti: {values['n_documents']}")

        for k in KS:
            print(
                f"  Ranking overlap@{k}: "
                f"{values[f'ranking_overlap@{k}']:.8f}"
            )

        print(
            f"  MRR reference@{MRR_K}: "
            f"{values[f'mrr_reference@{MRR_K}']:.8f}"
        )

    print("\nFile generati:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()