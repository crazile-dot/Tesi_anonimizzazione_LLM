import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_BASELINE_FILE = (
    BASE_DIR / "data" / "processed" / "embeddings_100.jsonl"
)

EMBEDDINGS_M4_FILE = (
    BASE_DIR / "data" / "processed" / "embeddings_m4_100.jsonl"
)

DATASET_FILE = (
    BASE_DIR / "data" / "processed" / "dataset_100_baseline.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" / "similarity_m4_100.jsonl"
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

    return float(
        np.dot(a, b) / (norm_a * norm_b)
    )


# ============================================================
# STATISTICHE
# ============================================================

def print_statistics(values):

    print("STATISTICHE")
    print("----------------------------------------")

    print(f"N:       {len(values)}")
    print(f"Media:   {np.mean(values):.8f}")
    print(f"Mediana: {np.median(values):.8f}")
    print(f"Std:     {np.std(values):.8f}")
    print(f"Varianza:{np.var(values):.8f}")
    print(f"Min:     {np.min(values):.8f}")
    print(f"Max:     {np.max(values):.8f}")
    print(f"Q1:      {np.percentile(values, 25):.8f}")
    print(f"Q3:      {np.percentile(values, 75):.8f}")
    print(
        f"IQR:     "
        f"{np.percentile(values, 75) - np.percentile(values, 25):.8f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"Lettura embedding baseline: "
        f"{EMBEDDINGS_BASELINE_FILE}"
    )

    print(
        f"Lettura embedding M4: "
        f"{EMBEDDINGS_M4_FILE}"
    )

    print()

    # --------------------------------------------------------
    # Caricamento ground truth
    # --------------------------------------------------------

    ground_truth = {}

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            document = json.loads(line)

            ground_truth[
                document["document_id"]
            ] = document["entities"]

    # --------------------------------------------------------
    # Caricamento embedding baseline
    # --------------------------------------------------------

    baseline_embeddings = {}

    with open(
        EMBEDDINGS_BASELINE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            document = json.loads(line)

            baseline_embeddings[
                document["document_id"]
            ] = document["embeddings"]["M0_original"]["embedding"]

    # --------------------------------------------------------
    # Calcolo similarity M0 vs M4
    # --------------------------------------------------------

    results = []
    excluded = 0
    missing_baseline = 0

    with open(
        EMBEDDINGS_M4_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            document = json.loads(line)

            document_id = document["document_id"]

            entities = ground_truth.get(
                document_id,
                []
            )

            # Escludiamo documenti senza PERSON
            if len(entities) == 0:
                excluded += 1
                continue

            # Controllo presenza embedding M0
            if document_id not in baseline_embeddings:

                print(
                    f"Embedding M0 non trovato: "
                    f"{document_id}"
                )

                missing_baseline += 1
                continue

            m0 = baseline_embeddings[
                document_id
            ]

            m4 = document[
                "embeddings"
            ][
                "M4_llm_substitution"
            ][
                "embedding"
            ]

            similarity = cosine_similarity(
                m0,
                m4
            )

            result = {
                "document_id": document_id,
                "M0_vs_M4": similarity
            }

            results.append(result)

            print(document_id)
            print(
                f"  M0 vs M4: "
                f"{similarity:.8f}"
            )
            print()

    # --------------------------------------------------------
    # Salvataggio risultati
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as out:

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
    print("RISULTATI M4")
    print("========================================")

    print(
        f"Documenti valutati: "
        f"{len(results)}"
    )

    print(
        f"Documenti esclusi (0 PERSON): "
        f"{excluded}"
    )

    print(
        f"Embedding M0 mancanti: "
        f"{missing_baseline}"
    )

    print()

    if results:

        values = [
            r["M0_vs_M4"]
            for r in results
            if r["M0_vs_M4"] is not None
        ]

        if values:

            print_statistics(values)

    print()
    print(
        f"Risultati salvati in: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()