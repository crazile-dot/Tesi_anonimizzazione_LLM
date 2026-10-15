
from pathlib import Path
from collections import defaultdict
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_DIR / "data" / "processed" / "email_manifest.csv"

# Per ogni nome, memorizziamo i documenti distinti
# e gli split nei quali compare.
documenti_per_persona = defaultdict(set)
split_per_persona = defaultdict(set)

with MANIFEST.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:
    reader = csv.DictReader(f)

    for row in reader:
        split = row["split"]
        document_id = row["document_id"]

        nomi = [
            nome.strip()
            for nome in row["annotated_names"].split("|")
            if nome.strip()
        ]

        for nome in nomi:
            documenti_per_persona[nome].add(document_id)
            split_per_persona[nome].add(split)

print(f"Varianti testuali di nomi: {len(documenti_per_persona)}")

# Distribuzione: quante varianti hanno 1, 2, 3... documenti
distribuzione = defaultdict(int)

for nome, documenti in documenti_per_persona.items():
    distribuzione[len(documenti)] += 1

print("\n--- Numero di documenti per variante del nome ---")
for numero in sorted(distribuzione):
    print(f"{numero} documento/i: {distribuzione[numero]} nomi")

print("\n--- Esempi di nomi presenti in più documenti ---")
persone_ordinate = sorted(
    documenti_per_persona.items(),
    key=lambda item: (-len(item[1]), item[0].casefold())
)

for nome, documenti in persone_ordinate[:30]:
    split = ", ".join(sorted(split_per_persona[nome]))
    print(
        f"{nome!r}: {len(documenti)} documenti "
        f"(split: {split})"
    )

# Conteggio delle varianti che compaiono sia in train sia in test.
con_train_e_test = sum(
    {"train", "test"}.issubset(split_per_persona[nome])
    for nome in documenti_per_persona
)

print(
    "\nVarianti testuali presenti sia in train sia in test: "
    f"{con_train_e_test}"
)
