
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "embeddings_680_all_v1.jsonl"
)
OUTPUT_DIR = BASE_DIR / "data" / "processed"

METHODS = [
    "M1_redaction",
    "M2_pseudonymization",
    "M3_hmac",
    "M4_llm_substitution",
    "M5_llm_anonymization",
]


def valid_embedding(item):
    if not isinstance(item, dict):
        return None

    if item.get("status") != "success":
        return None

    vector = item.get("embedding")
    if not isinstance(vector, list) or not vector:
        return None

    array = np.asarray(vector, dtype=float)

    if array.ndim != 1 or not np.all(np.isfinite(array)):
        return None

    norm = np.linalg.norm(array)
    if not np.isclose(norm, 1.0, atol=1e-4):
        return None

    return array


def cosine_similarity(a, b):
    if a.shape != b.shape:
        raise ValueError(
            f"Dimensioni incompatibili: {a.shape} e {b.shape}"
        )

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    ids = [r["document_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Sono presenti document_id duplicati.")

    rows = []

    for record in records:
        document_id = record["document_id"]
        embeddings = record.get("embeddings", {})
        original = valid_embedding(embeddings.get("M0_original"))

        row = {"document_id": document_id}

        if original is None:
            for method in METHODS:
                row[method] = np.nan
            rows.append(row)
            continue

        for method in METHODS:
            anonymized = valid_embedding(embeddings.get(method))

            if anonymized is None:
                row[method] = np.nan
            else:
                row[method] = cosine_similarity(original, anonymized)

        rows.append(row)

    per_document = pd.DataFrame(rows)
    per_document.to_csv(
        OUTPUT_DIR / "utility_cosine_per_document.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []

    for method in METHODS:
        values = per_document[method].dropna()

        summary_rows.append({
            "strategy": method,
            "n_documents": int(values.count()),
            "mean_cosine": values.mean(),
            "std_cosine": values.std(),
            "median_cosine": values.median(),
            "min_cosine": values.min(),
            "max_cosine": values.max(),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "utility_cosine_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Documenti nel dataset:", len(records))
    print("\n--- Similarità coseno rispetto a M0 ---")
    print(summary.round(4).to_string(index=False))
    print("\nFile salvati:")
    print(OUTPUT_DIR / "utility_cosine_per_document.csv")
    print(OUTPUT_DIR / "utility_cosine_summary.csv")


if __name__ == "__main__":
    main()
