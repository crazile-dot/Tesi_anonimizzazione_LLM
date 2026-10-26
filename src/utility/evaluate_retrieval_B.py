
'''import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = DATA_DIR / "embeddings_100.jsonl"

OUTPUT_PER_DOCUMENT = (
    DATA_DIR / "semantic_retrieval_anonymized_query_per_document_100.csv"
)
OUTPUT_SUMMARY = (
    DATA_DIR / "semantic_retrieval_anonymized_query_summary_100.csv"
)

METHODS = [
    "M0_original",
    "M1_redaction",
    "M2_pseudonymization",
    "M3_hmac",
]

K_VALUES = [5, 10]
MRR_K = 10
NDCG_K = 10


# ============================================================
# CARICAMENTO
# ============================================================

def load_jsonl(filepath):
    records = {}

    with filepath.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            document_id = record["document_id"]

            if document_id in records:
                raise ValueError(
                    f"ID duplicato alla riga {line_number}: "
                    f"{document_id}"
                )

            records[document_id] = record

    return records


def get_embedding(record, method):
    try:
        vector = record["embeddings"][method]["embedding"]
    except (KeyError, TypeError):
        return None

    if not isinstance(vector, list) or not vector:
        return None

    vector = np.asarray(vector, dtype=np.float64)

    if vector.ndim != 1 or not np.isfinite(vector).all():
        return None

    if np.linalg.norm(vector) == 0:
        return None

    return vector


# ============================================================
# SIMILARITA' E RANKING
# ============================================================

def similarity_matrix(embeddings):
    matrix = np.asarray(embeddings, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError("Gli embedding devono formare una matrice 2D.")

    norms = np.linalg.norm(matrix, axis=1)

    if np.any(norms == 0) or not np.isfinite(matrix).all():
        raise ValueError("Embedding non validi.")

    matrix = matrix / norms[:, None]
    return matrix @ matrix.T


def get_ranking(matrix, query_index):
    scores = matrix[query_index].copy()

    # Escludi la stessa email dai risultati.
    scores[query_index] = -np.inf

    # Ordinamento deterministico in caso di parità.
    return np.argsort(-scores, kind="stable")


# ============================================================
# METRICHE
# ============================================================

def recall_at_k(reference, candidate, k):
    reference_set = set(reference[:k])
    candidate_set = set(candidate[:k])

    return len(reference_set & candidate_set) / k


def mrr_reference(reference, candidate, k):
    relevant = set(reference[:k])

    for rank, document_index in enumerate(candidate[:k], start=1):
        if document_index in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_reference(reference, candidate, k):
    # Rilevanza graduata in base alla posizione nel ranking M0:
    # posizione 1 = rilevanza k, posizione k = rilevanza 1.
    relevance = {
        document_index: k - rank
        for rank, document_index in enumerate(reference[:k])
    }

    dcg = 0.0

    for rank, document_index in enumerate(candidate[:k], start=1):
        rel = relevance.get(document_index, 0)
        dcg += rel / np.log2(rank + 1)

    ideal_relevances = sorted(relevance.values(), reverse=True)

    ideal_dcg = sum(
        rel / np.log2(rank + 1)
        for rank, rel in enumerate(ideal_relevances, start=1)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0


# ============================================================
# ESPERIMENTO B
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File non trovato: {INPUT_FILE}"
        )

    records = load_jsonl(INPUT_FILE)

    first_id = next(iter(records))
    first_record = records[first_id]

    print("\nDocumento di esempio:", first_id)

    for method, value in first_record["embeddings"].items():
        print(f"\nMetodo: {method}")
        print("Tipo:", type(value).__name__)

        if isinstance(value, dict):
            print("Chiavi:", list(value.keys()))

            for key, item in value.items():
                if isinstance(item, list):
                    print(f"  {key}: lista di lunghezza {len(item)}")
                else:
                    print(f"  {key}: {type(item).__name__}")
        elif isinstance(value, list):
            print("Lunghezza:", len(value))

    # Usiamo il medesimo insieme di documenti per tutte le strategie.
    # Questo esclude il documento con embedding M4/M5 non riuscito.
    document_ids = []

    for document_id, record in records.items():
        if all(
            get_embedding(record, method) is not None
            for method in METHODS
        ):
            document_ids.append(document_id)

    document_ids.sort()

    if len(document_ids) < 11:
        raise ValueError("Documenti validi insufficienti.")

    print(f"Documenti nel file: {len(records)}")
    print(f"Documenti validi comuni: {len(document_ids)}")

    embeddings = {}

    for method in METHODS:
        embeddings[method] = np.stack([
            get_embedding(records[doc_id], method)
            for doc_id in document_ids
        ])

    # Ranking originale: riferimento invariato per tutti i metodi.
    reference_matrix = similarity_matrix(embeddings["M0_original"])
    reference_rankings = [
        get_ranking(reference_matrix, i)
        for i in range(len(document_ids))
    ]

    per_document = []

    for method in METHODS:
        print(f"Valutazione: {method}")

        # ESPERIMENTO B:
        # sia query sia candidati provengono dalla stessa strategia.
        method_matrix = similarity_matrix(embeddings[method])

        for i, document_id in enumerate(document_ids):
            candidate_ranking = get_ranking(method_matrix, i)
            reference_ranking = reference_rankings[i]

            result = {
                "document_id": document_id,
                "strategy": method,
            }

            for k in K_VALUES:
                result[f"Recall@{k}"] = recall_at_k(
                    reference_ranking, candidate_ranking, k
                )

            result[f"MRR@{MRR_K}"] = mrr_reference(
                reference_ranking, candidate_ranking, MRR_K
            )

            result[f"nDCG@{NDCG_K}"] = ndcg_reference(
                reference_ranking, candidate_ranking, NDCG_K
            )

            per_document.append(result)

    df = pd.DataFrame(per_document)
    df.to_csv(OUTPUT_PER_DOCUMENT, index=False, encoding="utf-8-sig")

    metric_columns = [
        "Recall@5",
        "Recall@10",
        "MRR@10",
        "nDCG@10",
    ]

    summary = (
        df.groupby("strategy")[metric_columns]
        .agg(["mean", "std"])
        .round(6)
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()
    summary.insert(1, "n_queries", len(document_ids))
    summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n=== ESPERIMENTO B: QUERY E INDICE TRASFORMATI ===")
    print(summary.to_string(index=False))

    print("\nFile generati:")
    print(OUTPUT_PER_DOCUMENT)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()


import json
from pathlib import Path

files = [
    Path("data/processed/embeddings_100.jsonl"),
    Path("data/processed/embeddings_680_all_v1.jsonl"),
]

for file in files:
    print(f"\n{'=' * 60}")
    print(f"FILE: {file}")

    with file.open("r", encoding="utf-8") as f:
        first_record = json.loads(next(f))

    print("Chiavi principali:", list(first_record.keys()))

    for key, value in first_record.items():
        if isinstance(value, dict):
            print(f"  {key}: dict, chiavi = {list(value.keys())}")
        elif isinstance(value, list):
            print(f"  {key}: list, lunghezza = {len(value)}")
            if value:
                print(f"    Primo elemento: {str(value[0])[:200]}")
        else:
            print(f"  {key}: {type(value).__name__} = {str(value)[:200]}")
'''



# CONFRONTO METADATI

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

FILE_A = DATA_DIR / "embeddings_100.jsonl"
FILE_B = DATA_DIR / "embeddings_680_all_v1.jsonl"

OUT_DOC = DATA_DIR / "confronto_metadata_100_per_documento.csv"
OUT_SUM = DATA_DIR / "confronto_metadata_100_summary.csv"

# Configurazione A: solo nomi anonimizzati.
METHODS_A = {
    "Redaction": "M1_redaction",
    "Pseudonymization": "M2_pseudonymization",
    "HMAC": "M3_hmac",
}

# Nomi possibili dei metodi nel file esteso.
ALIASES_B = {
    "Original": ["M0_original", "M0_originale"],
    "Redaction": ["M1_redacted", "M1_redaction"],
    "Pseudonymization": ["M2_persona", "M2_pseudonymization"],
    "HMAC": ["M3_hmac"],
}


def load_jsonl(path):
    data = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                data[r["document_id"]] = r
    return data


def get_vector(record, method):
    try:
        v = record["embeddings"][method]["embedding"]
        v = np.asarray(v, dtype=float)
        if v.ndim == 1 and len(v) and np.isfinite(v).all():
            if np.linalg.norm(v) > 0:
                return v
    except (KeyError, TypeError, ValueError):
        pass
    return None


def find_method(record, aliases):
    for name in aliases:
        if get_vector(record, name) is not None:
            return name
    return None


def cosine_matrix(vectors):
    x = np.stack(vectors)
    norms = np.linalg.norm(x, axis=1)
    x = x / norms[:, None]
    return x @ x.T


def rankings(matrix):
    result = []
    for i in range(len(matrix)):
        scores = matrix[i].copy()
        scores[i] = -np.inf
        result.append(np.argsort(-scores, kind="stable"))
    return result


def recall(ref, cand, k):
    return len(set(ref[:k]) & set(cand[:k])) / k


def mrr(ref, cand, k=10):
    relevant = set(ref[:k])
    for rank, idx in enumerate(cand[:k], start=1):
        if idx in relevant:
            return 1 / rank
    return 0.0


def ndcg(ref, cand, k=10):
    relevance = {idx: k - rank for rank, idx in enumerate(ref[:k])}
    dcg = sum(
        relevance.get(idx, 0) / np.log2(rank + 1)
        for rank, idx in enumerate(cand[:k], start=1)
    )
    ideal = sum(
        rel / np.log2(rank + 1)
        for rank, rel in enumerate(
            sorted(relevance.values(), reverse=True), start=1
        )
    )
    return dcg / ideal if ideal else 0.0


import numpy as np

def diagnostica_embedding(record_a, record_b, metodo_a, metodo_b):
    """Confronta gli embedding dello stesso documento tra due pipeline."""

    emb_a = record_a["embeddings"][metodo_a]
    emb_b = record_b["embeddings"][metodo_b]

    v_a = np.asarray(emb_a["embedding"], dtype=np.float64)
    v_b = np.asarray(emb_b["embedding"], dtype=np.float64)

    print(f"\nDocumento: {record_a['document_id']}")
    print(f"Metodo A: {metodo_a}")
    print(f"Metodo B: {metodo_b}")
    print(f"Dimensioni: A={v_a.shape}, B={v_b.shape}")
    print(f"Chunk: A={emb_a.get('num_chunks')}, "
          f"B={emb_b.get('num_chunks')}")

    if v_a.shape != v_b.shape:
        print("ERRORE: dimensioni differenti")
        return

    cos_sim = np.dot(v_a, v_b) / (
        np.linalg.norm(v_a) * np.linalg.norm(v_b)
    )

    print(f"Cosine similarity: {cos_sim:.10f}")
    print(f"Differenza assoluta massima: "
          f"{np.max(np.abs(v_a - v_b)):.10e}")
    print(f"Norma A: {np.linalg.norm(v_a):.10f}")
    print(f"Norma B: {np.linalg.norm(v_b):.10f}")
    print(f"Allclose (atol=1e-5): "
          f"{np.allclose(v_a, v_b, atol=1e-5)}")


def main():
    for path in [FILE_A, FILE_B]:
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")

    a = load_jsonl(FILE_A)
    b = load_jsonl(FILE_B)

    common = sorted(set(a) & set(b))
    valid = []

    method_names_b = {}
    for label, aliases in ALIASES_B.items():
        method_names_b[label] = find_method(next(iter(b.values())), aliases)

    print("Metodi trovati nel file esteso:", method_names_b)

    if any(v is None for v in method_names_b.values()):
        

        raise ValueError(
            "Nomi dei metodi non riconosciuti nel file esteso. "
            "Controlla le chiavi di embeddings_680_all_v1.jsonl."
        )

    for doc_id in common:
        ok_a = all(
            get_vector(a[doc_id], method) is not None
            for method in METHODS_A.values()
        )
        ok_b = all(
            get_vector(b[doc_id], method) is not None
            for method in method_names_b.values()
        )
        if ok_a and ok_b:
            valid.append(doc_id)

    if len(valid) < 11:
        raise ValueError(f"Documenti comuni validi insufficienti: {len(valid)}")

    print("Documenti comuni:", len(common))
    print("Documenti validi usati:", len(valid))

    # Ranking di riferimento: embedding originali.
    # Verifichiamo anche che gli originali dei due file coincidano.
    original_a = np.stack([
        get_vector(a[d], "M0_original") for d in valid
    ])
    original_b = np.stack([
        get_vector(b[d], method_names_b["Original"]) for d in valid
    ])

    
    for i, (x, y) in enumerate(zip(original_a, original_b)):
        if x.shape != y.shape or not np.allclose(x, y, atol=1e-5):
            doc_id = valid[i]

            print("\n>>> DIAGNOSTICA EMBEDDING ORIGINALI <<<")
            print(f"Documento: {doc_id}")
            print(f"Dimensioni A: {x.shape}")
            print(f"Dimensioni B: {y.shape}")

            if x.shape == y.shape:
                cosine = np.dot(x, y) / (
                    np.linalg.norm(x) * np.linalg.norm(y)
                )
                print(f"Cosine similarity: {cosine:.10f}")
                print(
                    "Differenza assoluta massima:",
                    f"{np.max(np.abs(x - y)):.10e}"
                )

                emb_a = a[doc_id]["embeddings"]["M0_original"]
                emb_b = b[doc_id]["embeddings"][method_names_b["Original"]]
                print(f"Chunk A: {emb_a.get('num_chunks')}")
                print(f"Chunk B: {emb_b.get('num_chunks')}")

            raise ValueError(
                "Embedding originali diversi: controllare la diagnostica."
            )


    ref = rankings(cosine_matrix(original_a))
    rows = []

    for label in METHODS_A:
        vectors_a = np.stack([
            get_vector(a[d], METHODS_A[label]) for d in valid
        ])
        vectors_b = np.stack([
            get_vector(b[d], method_names_b[label]) for d in valid
        ])

        for config, vectors in [
            ("A_soli_nomi", vectors_a),
            ("B_nomi_email_message_id", vectors_b),
        ]:
            cand_rankings = rankings(cosine_matrix(vectors))

            for i, doc_id in enumerate(valid):
                rr = ref[i]
                cr = cand_rankings[i]
                rows.append({
                    "document_id": doc_id,
                    "configurazione": config,
                    "strategia": label,
                    "Recall@5": recall(rr, cr, 5),
                    "Recall@10": recall(rr, cr, 10),
                    "MRR@10": mrr(rr, cr),
                    "nDCG@10": ndcg(rr, cr),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DOC, index=False, encoding="utf-8-sig")

    metrics = ["Recall@5", "Recall@10", "MRR@10", "nDCG@10"]
    summary = (
        df.groupby(["strategia", "configurazione"])[metrics]
        .mean()
        .reset_index()
    )

    # Differenza B - A: valori negativi indicano una diminuzione.
    wide = summary.pivot(
        index="strategia", columns="configurazione", values=metrics
    )

    comparison = []
    for strategy in METHODS_A:
        row = {"strategia": strategy}
        for metric in metrics:
            av = wide.loc[strategy, (metric, "A_soli_nomi")]
            bv = wide.loc[strategy, (metric, "B_nomi_email_message_id")]
            row[f"{metric}_A"] = av
            row[f"{metric}_B"] = bv
            row[f"{metric}_delta_B_meno_A"] = bv - av
        comparison.append(row)

    result = pd.DataFrame(comparison).round(6)
    result.to_csv(OUT_SUM, index=False, encoding="utf-8-sig")

    print("\n=== CONFRONTO A/B SUGLI STESSI DOCUMENTI ===")
    print(result.to_string(index=False))
    print("\nFile generati:")
    print(OUT_DOC)
    print(OUT_SUM)


if __name__ == "__main__":
    main()
