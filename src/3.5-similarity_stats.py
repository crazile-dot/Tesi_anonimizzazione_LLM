import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "similarity_100.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "similarity_summary.json"


def calculate_stats(values):
    values = np.array(values)

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "variance": float(np.var(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1)
    }


def main():
    print(f"Lettura risultati: {INPUT_FILE}")
    print()

    results = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"Documenti valutati: {len(results)}")
    print()

    m1_values = [r["M0_vs_M1"] for r in results]
    m2_values = [r["M0_vs_M2"] for r in results]
    m3_values = [r["M0_vs_M3"] for r in results]

    summary = {
        "M0_vs_M1": calculate_stats(m1_values),
        "M0_vs_M2": calculate_stats(m2_values),
        "M0_vs_M3": calculate_stats(m3_values)
    }

    print("========================================")
    print("STATISTICHE COSINE SIMILARITY")
    print("========================================")

    for method, stats in summary.items():
        print()
        print(method)
        print("----------------------------------------")
        print(f"N:          {stats['n']}")
        print(f"Media:      {stats['mean']:.8f}")
        print(f"Mediana:    {stats['median']:.8f}")
        print(f"Dev. std.:  {stats['std']:.8f}")
        print(f"Variance : {stats['variance']:.8f}")
        print(f"Minimo:     {stats['min']:.8f}")
        print(f"Massimo:    {stats['max']:.8f}")
        print(f"Q1:         {stats['q1']:.8f}")
        print(f"Q3:         {stats['q3']:.8f}")
        print(f"IQR:        {stats['iqr']:.8f}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("========================================")
    print(f"Statistiche salvate in: {OUTPUT_FILE}")
    print("========================================")


if __name__ == "__main__":
    main()