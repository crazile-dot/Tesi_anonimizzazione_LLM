import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_FILE = (
    BASE_DIR / "data" / "processed" / "embeddings_100.jsonl"
)

DATASET_FILE = (
    BASE_DIR / "data" / "processed" / "dataset_100_baseline.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" / "similarity_100.jsonl"
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):
    """
    Calcola la cosine similarity tra due vettori.
    """

    a = np.array(vector_a)
    b = np.array(vector_b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return None

    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================
# MAIN
# ============================================================

def main():

    print(f"Lettura dataset: {DATASET_FILE}")
    print(f"Lettura embedding: {EMBEDDINGS_FILE}")
    print()

    # --------------------------------------------------------
    # Caricamento ground truth
    # --------------------------------------------------------

    ground_truth = {}

    with open(DATASET_FILE, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            document = json.loads(line)

            ground_truth[document["document_id"]] = document["entities"]

    # --------------------------------------------------------
    # Calcolo similarity
    # --------------------------------------------------------

    results = []
    excluded = 0

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            document = json.loads(line)

            document_id = document["document_id"]

            entities = ground_truth.get(document_id, [])

            # Escludiamo i documenti senza PERSON
            if len(entities) == 0:
                excluded += 1
                continue

            embeddings = document["embeddings"]

            m0 = embeddings["M0_original"]["embedding"]

            similarity_m1 = cosine_similarity(
                m0,
                embeddings["M1_redaction"]["embedding"]
            )

            similarity_m2 = cosine_similarity(
                m0,
                embeddings["M2_pseudonymization"]["embedding"]
            )

            similarity_m3 = cosine_similarity(
                m0,
                embeddings["M3_hmac"]["embedding"]
            )

            result = {
                "document_id": document_id,
                "M0_vs_M1": similarity_m1,
                "M0_vs_M2": similarity_m2,
                "M0_vs_M3": similarity_m3
            }

            results.append(result)

            print(document_id)
            print(f"  M0 vs M1: {similarity_m1:.8f}")
            print(f"  M0 vs M2: {similarity_m2:.8f}")
            print(f"  M0 vs M3: {similarity_m3:.8f}")
            print()

    # --------------------------------------------------------
    # Salvataggio risultati
    # --------------------------------------------------------

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for result in results:

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

    # --------------------------------------------------------
    # Statistiche
    # --------------------------------------------------------

    print("========================================")
    print("RISULTATI")
    print("========================================")

    print(f"Documenti valutati: {len(results)}")
    print(f"Documenti esclusi (0 PERSON): {excluded}")
    print()

    if results:

        m1_values = [r["M0_vs_M1"] for r in results]
        m2_values = [r["M0_vs_M2"] for r in results]
        m3_values = [r["M0_vs_M3"] for r in results]

        print("MEDIA COSINE SIMILARITY")
        print("----------------------------------------")

        print(f"M0 vs M1: {np.mean(m1_values):.8f}")
        print(f"M0 vs M2: {np.mean(m2_values):.8f}")
        print(f"M0 vs M3: {np.mean(m3_values):.8f}")

    print()
    print(f"Risultati salvati in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()