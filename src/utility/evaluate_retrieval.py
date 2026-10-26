
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "embeddings_680_all_v1.jsonl"
)
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_PER_DOCUMENT = OUTPUT_DIR / "semantic_retrieval_per_document.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "semantic_retrieval_summary.csv"

EMBEDDING_KEYS = {
    "M0": "M0_original",
    "M1": "M1_redaction",
    "M2": "M2_pseudonymization",
    "M3": "M3_hmac",
    "M4": "M4_llm_substitution",
    "M5": "M5_llm_anonymization",
}

K_VALUES = (5, 10)


# ============================================================
# FUNZIONI
# ============================================================

def ranking(scores, query_index):
    """Ordina i documenti per similarità, escludendo la query stessa."""
    scores = scores.copy()
    scores[query_index] = -np.inf
    return np.argsort(-scores, kind="stable")


def evaluate_ranking(reference_ranking, candidate_ranking):
    """
    Confronta un ranking con il ranking di riferimento M0.

    Recall@k:
        quota dei top-k di M0 presenti nei top-k della strategia.

    MRR@10:
        reciproco della posizione del primo documento appartenente
        ai top-10 di M0.

    nDCG@10:
        usa rilevanza graduata derivata dalla posizione nel top-10
        di M0: rilevanza 10 per il primo, ..., 1 per il decimo.
        Gli altri documenti hanno rilevanza zero.
    """
    metrics = {}

    reference_top = {
        k: set(reference_ranking[:k])
        for k in K_VALUES
    }

    for k in K_VALUES:
        candidate_top = set(candidate_ranking[:k])
        metrics[f"Recall@{k}"] = (
            len(reference_top[k] & candidate_top) / k
        )

    # MRR@10: primo risultato appartenente ai top-10 di M0.
    relevant = reference_top[10]
    reciprocal_rank = 0.0

    for position, doc_index in enumerate(candidate_ranking[:10], start=1):
        if doc_index in relevant:
            reciprocal_rank = 1.0 / position
            break

    metrics["MRR@10"] = reciprocal_rank

    # nDCG@10: rilevanza graduata secondo il ranking M0.
    reference_position = {
        doc_index: position
        for position, doc_index in enumerate(reference_ranking[:10], start=1)
    }

    gains = []
    for doc_index in candidate_ranking[:10]:
        reference_rank = reference_position.get(doc_index)

        if reference_rank is None:
            relevance = 0.0
        else:
            relevance = float(11 - reference_rank)

        gains.append(relevance)

    discounts = np.log2(np.arange(2, 12))
    dcg = float(np.sum(np.asarray(gains) / discounts))

    ideal_gains = np.arange(10, 0, -1, dtype=float)
    idcg = float(np.sum(ideal_gains / discounts))

    metrics["nDCG@10"] = dcg / idcg if idcg > 0 else 0.0

    return metrics


# ============================================================
# LETTURA DEGLI EMBEDDING
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"File non trovato: {INPUT_FILE}")

records = []

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
        if not line.strip():
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSON non valido alla riga {line_number}"
            ) from exc

if not records:
    raise ValueError("Il file degli embedding è vuoto.")

document_ids = [record.get("document_id") for record in records]

if any(doc_id is None for doc_id in document_ids):
    raise ValueError("Alcuni record non hanno document_id.")

if len(set(document_ids)) != len(document_ids):
    raise ValueError("Sono presenti document_id duplicati.")

# Manteniamo solo documenti con embedding validi per tutte le strategie.
valid_records = []

for record in records:
    embeddings = record.get("embeddings", {})
    valid = True

    for strategy, key in EMBEDDING_KEYS.items():
        item = embeddings.get(key, {})
        vector = item.get("embedding")

        if item.get("status") != "success" or vector is None:
            valid = False
            break

        vector = np.asarray(vector, dtype=float)

        if vector.ndim != 1 or vector.size == 0:
            valid = False
            break

        if not np.isfinite(vector).all():
            valid = False
            break

        norm = np.linalg.norm(vector)
        if norm <= 0:
            valid = False
            break

    if valid:
        valid_records.append(record)

if len(valid_records) < 11:
    raise ValueError(
        "Servono almeno 11 documenti validi per calcolare le metriche @10."
    )

ids = [record["document_id"] for record in valid_records]
n_documents = len(ids)

print(f"Record nel file originale: {len(records)}")
print(f"Documenti con tutti gli embedding validi: {n_documents}")


# ============================================================
# COSTRUZIONE DELLE MATRICI
# ============================================================

matrices = {}

for strategy, key in EMBEDDING_KEYS.items():
    vectors = np.asarray(
        [
            record["embeddings"][key]["embedding"]
            for record in valid_records
        ],
        dtype=float,
    )

    if vectors.ndim != 2:
        raise ValueError(f"Matrice non valida per {strategy}.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    # Normalizzazione esplicita per calcolare la similarità coseno.
    vectors = vectors / norms

    matrices[strategy] = vectors

dimensions = {matrix.shape[1] for matrix in matrices.values()}

if len(dimensions) != 1:
    raise ValueError("Le strategie hanno dimensioni embedding differenti.")

# Per ogni strategia, matrice query originale M0 × indice della strategia.
# Le righe sono le query; le colonne sono i documenti candidati.
scores_by_strategy = {}

for strategy, matrix in matrices.items():
    scores_by_strategy[strategy] = matrices["M0"] @ matrix.T


# ============================================================
# RANKING DI RIFERIMENTO M0
# ============================================================

reference_rankings = []

for query_index in range(n_documents):
    reference_rankings.append(
        ranking(scores_by_strategy["M0"][query_index], query_index)
    )


# ============================================================
# VALUTAZIONE PER DOCUMENTO E STRATEGIA
# ============================================================

rows = []

for query_index, query_id in enumerate(ids):
    reference = reference_rankings[query_index]

    for strategy in EMBEDDING_KEYS:
        candidate = ranking(
            scores_by_strategy[strategy][query_index],
            query_index,
        )

        metrics = evaluate_ranking(reference, candidate)

        row = {
            "query_document_id": query_id,
            "strategy": strategy,
            "n_candidates": n_documents - 1,
        }
        row.update(metrics)
        rows.append(row)

per_document = pd.DataFrame(rows)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
per_document.to_csv(
    OUTPUT_PER_DOCUMENT,
    index=False,
    float_format="%.8f",
)


# ============================================================
# RIEPILOGO DESCRITTIVO
# ============================================================

metric_columns = [
    "Recall@5",
    "Recall@10",
    "MRR@10",
    "nDCG@10",
]

summary_rows = []

for strategy in EMBEDDING_KEYS:
    subset = per_document.loc[
        per_document["strategy"] == strategy
    ]

    row = {
        "strategy": strategy,
        "n_queries": len(subset),
    }

    for metric in metric_columns:
        values = subset[metric]

        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std(ddof=1)
        row[f"{metric}_median"] = values.median()
        row[f"{metric}_Q1"] = values.quantile(0.25)
        row[f"{metric}_Q3"] = values.quantile(0.75)

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    float_format="%.8f",
)

print("\n=== SEMANTIC RETRIEVAL: RIEPILOGO ===")
display_columns = ["strategy", "n_queries"]

for metric in metric_columns:
    display_columns.extend([
        f"{metric}_mean",
        f"{metric}_std",
    ])

print(summary[display_columns].round(4).to_string(index=False))

print(f"\nRisultati per documento: {OUTPUT_PER_DOCUMENT}")
print(f"Riepilogo: {OUTPUT_SUMMARY}")
print("\nValutazione completata.")
