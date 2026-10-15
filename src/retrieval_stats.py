import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "retrieval_100.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "retrieval_summary.json"

METHODS = [
    "M1_redaction",
    "M2_pseudonymization",
    "M3_hmac",
]

METRICS = [
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "mrr@5",
]


def calculate_statistics(values):
    values = np.array(values, dtype=float)

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "variance": float(np.var(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "iqr": float(
            np.percentile(values, 75)
            - np.percentile(values, 25)
        ),
    }


# ---------------------------------------------------------
# Caricamento risultati
# ---------------------------------------------------------

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))


print(f"Risultati caricati: {len(records)}")


# ---------------------------------------------------------
# Calcolo statistiche
# ---------------------------------------------------------

summary = {}


for method in METHODS:

    method_records = [
        record
        for record in records
        if record["method"] == method
    ]

    summary[method] = {}

    print(f"\n{method}")

    for metric in METRICS:

        values = [
            record[metric]
            for record in method_records
        ]

        stats = calculate_statistics(values)

        summary[method][metric] = stats

        print(f"\n  {metric}")
        print(f"    N        : {stats['n']}")
        print(f"    Mean     : {stats['mean']:.8f}")
        print(f"    Median   : {stats['median']:.8f}")
        print(f"    Variance : {stats['variance']:.8f}")
        print(f"    Std      : {stats['std']:.8f}")
        print(f"    Min      : {stats['min']:.8f}")
        print(f"    Max      : {stats['max']:.8f}")
        print(f"    Q1       : {stats['q1']:.8f}")
        print(f"    Q3       : {stats['q3']:.8f}")
        print(f"    IQR      : {stats['iqr']:.8f}")


# ---------------------------------------------------------
# Salvataggio
# ---------------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\nStatistiche salvate in:")
print(OUTPUT_FILE)