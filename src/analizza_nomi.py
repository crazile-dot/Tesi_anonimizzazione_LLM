
from pathlib import Path
from collections import defaultdict
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_DIR / "data" / "processed" / "email_manifest.csv"

documenti = defaultdict(set)
split_per_nome = defaultdict(set)

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        percorso = row["relative_path"]
        split = row["split"]

        for nome in row["annotated_names"].split("|"):
            nome = " ".join(nome.split())
            if not nome:
                continue

            # Uniformiamo maiuscole/minuscole,
            # senza unire nomi diversi per supposizione.
            chiave = nome.casefold()
            documenti[chiave].add(percorso)
            split_per_nome[chiave].add(split)

condivisi = [
    nome for nome in documenti
    if {"train", "test"}.issubset(split_per_nome[nome])
]

composti = [
    nome for nome in condivisi
    if len(nome.split()) >= 2
]

composti_ordinati = sorted(
    composti,
    key=lambda nome: (-len(documenti[nome]), nome)
)

print(f"Nomi condivisi tra train e test: {len(condivisi)}")
print(f"Nomi composti condivisi: {len(composti)}")

print("\n--- Esempi di nomi composti condivisi ---")
for nome in composti_ordinati[:30]:
    print(
        f"{nome}: {len(documenti[nome])} documenti"
    )
