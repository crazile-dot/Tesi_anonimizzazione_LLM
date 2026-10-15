
from pathlib import Path
from collections import Counter
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_DIR / "data" / "processed" / "email_manifest.csv"

documenti_per_nome = {}
email_con_annotazioni = Counter()
email_totali = Counter()
nomi_per_tipo = Counter()

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        split = row["split"]
        email_totali[split] += 1

        nomi = {
            " ".join(nome.split())
            for nome in row["annotated_names"].split("|")
            if nome.strip()
        }

        if nomi:
            email_con_annotazioni[split] += 1

        for nome in nomi:
            chiave = nome.casefold()

            if chiave not in documenti_per_nome:
                documenti_per_nome[chiave] = {
                    "train": set(),
                    "test": set(),
                    "varianti": set(),
                }

            documenti_per_nome[chiave][split].add(
                row["relative_path"]
            )
            documenti_per_nome[chiave]["varianti"].add(nome)

print("--- EMAIL ---")
for split in ("train", "test"):
    totale = email_totali[split]
    annotate = email_con_annotazioni[split]
    percentuale = 100 * annotate / totale if totale else 0

    print(
        f"{split}: {totale} email totali; "
        f"{annotate} con nomi annotati "
        f"({percentuale:.1f}%)"
    )

print("\n--- NOMI CONDIVISI TRA TRAIN E TEST ---")
condivisi = [
    (nome, dati)
    for nome, dati in documenti_per_nome.items()
    if dati["train"] and dati["test"]
]

singoli = [
    nome for nome, _ in condivisi
    if len(nome.split()) == 1
]

composti = [
    nome for nome, _ in condivisi
    if len(nome.split()) >= 2
]

print(f"Nomi condivisi complessivi: {len(condivisi)}")
print(f"Nomi di una parola: {len(singoli)}")
print(f"Nomi di più parole: {len(composti)}")

print("\n--- VARIANTI TESTUALI POTENZIALMENTE DA VERIFICARE ---")
varianti = [
    (nome, dati)
    for nome, dati in documenti_per_nome.items()
    if len(dati["varianti"]) > 1
]

print(
    "Gruppi di grafie che differiscono solo per "
    f"maiuscole/minuscole: {len(varianti)}"
)

print("\n--- ESEMPI DI NOMI COMPOSTI CONDIVISI ---")
esempi = sorted(
    [
        (nome, dati)
        for nome, dati in condivisi
        if len(nome.split()) >= 2
    ],
    key=lambda elemento: (
        -len(elemento[1]["train"]),
        elemento[0]
    )
)

for nome, dati in esempi[:15]:
    print(
        f"{nome}: "
        f"{len(dati['train'])} email train, "
        f"{len(dati['test'])} email test"
    )
