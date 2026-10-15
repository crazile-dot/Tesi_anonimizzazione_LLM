
import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"


def leggi_similarity(nome_file, campo):
    risultati = {}

    with open(BASE / nome_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                risultati[record["document_id"]] = record[campo]

    return risultati


m4 = leggi_similarity("similarity_m4_100.jsonl", "M0_vs_M4")
m5 = leggi_similarity("similarity_m5_100.jsonl", "M0_vs_M5")

comuni = sorted(set(m4) & set(m5))

valori_m4 = np.array([m4[i] for i in comuni])
valori_m5 = np.array([m5[i] for i in comuni])

print(f"Documenti M4: {len(m4)}")
print(f"Documenti M5: {len(m5)}")
print(f"Documenti comuni: {len(comuni)}")
print()
print(f"M4 — media: {valori_m4.mean():.4f}")
print(f"M4 — mediana: {np.median(valori_m4):.4f}")
print()
print(f"M5 — media: {valori_m5.mean():.4f}")
print(f"M5 — mediana: {np.median(valori_m5):.4f}")
print()
print(
    "Differenza media M5 - M4: "
    f"{(valori_m5 - valori_m4).mean():.4f}"
)
