"""
sample_dataset.py

Scopo:
Selezionare un campione di 100 documenti dal corpus EnronRandom-XML
per le successive sperimentazioni sull'anonimizzazione e sulla
valutazione della privacy e dell'utilità semantica.

Funzionamento:
1. Esplora ricorsivamente il dataset originale, escludendo gli archivi
   ZIP.
2. Analizza ogni documento e calcola il numero di annotazioni
   PERSON racchiuse nei tag <true_name>...</true_name>, la lunghezza
   del testo e lo split di provenienza.
3. Suddivide i documenti in quattro fasce in base al numero di
   annotazioni PERSON: 0, 1-2, 3-10 e oltre 10.
4. Seleziona 100 documenti mediante campionamento stratificato
   con seed 42: 10 con zero annotazioni, 30 con 1-2, 40 con 3-10
   e 20 con oltre 10 annotazioni.
5. All'interno dell'ultima fascia, seleziona 15 documenti casualmente
   e aggiunge 5 documenti con il maggior numero di annotazioni,
   evitando duplicati.
6. Verifica che il campione contenga 100 documenti distinti e
   ordina i risultati per identificativo.
7. Salva il campione in formato CSV e stampa un riepilogo.

Input:
- data/raw/enronRandom-XML/

Output:
- data/processed/sample_100.csv

Nota:
Il campione combina una selezione casuale all'interno di fasce
predefinite con l'inclusione mirata di documenti ad alta densità
di annotazioni. Non costituisce quindi un campione puramente casuale
dell'intero corpus.
"""

from pathlib import Path
import re
import random
import csv


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_PATH
    / "data"
    / "raw"
    / "enronRandom-XML"
)

OUTPUT_PATH = (
    PROJECT_PATH
    / "data"
    / "processed"
)

SEED = 42


# ============================================================
# RICERCA DEI DOCUMENTI
# ============================================================

files = sorted([
    f
    for f in DATASET_PATH.rglob("*")
    if f.is_file() and f.suffix != ".zip"
])

print(f"Documenti trovati: {len(files)}")


# ============================================================
# ANALISI DEI DOCUMENTI
# ============================================================

pattern = r"<true_name>(.*?)</true_name>"

documents = []

for file_path in files:

    text = file_path.read_text(
        encoding="latin-1",
        errors="replace"
    )

    person_count = len(
        re.findall(
            pattern,
            text,
            flags=re.DOTALL
        )
    )

    length = len(text)

    # Determiniamo la fascia
    if person_count == 0:
        stratum = "0"

    elif person_count <= 2:
        stratum = "1-2"

    elif person_count <= 10:
        stratum = "3-10"

    else:
        stratum = ">10"

    documents.append({
        "document_id": file_path.name,
        "path": str(file_path),
        "split": file_path.relative_to(DATASET_PATH).parts[0],
        "person_count": person_count,
        "length": length,
        "stratum": stratum
    })


# ============================================================
# DIVISIONE PER STRATO
# ============================================================

strata = {
    "0": [],
    "1-2": [],
    "3-10": [],
    ">10": []
}

for document in documents:
    strata[document["stratum"]].append(document)


print("\nDocumenti per fascia:")

for stratum, docs in strata.items():
    print(f"{stratum}: {len(docs)}")


# ============================================================
# CAMPIONAMENTO
# ============================================================

random.seed(SEED)

sample = []

# 0 PERSON → 10
sample.extend(
    random.sample(strata["0"], 10)
)

# 1-2 PERSON → 30
sample.extend(
    random.sample(strata["1-2"], 30)
)

# 3-10 PERSON → 40
sample.extend(
    random.sample(strata["3-10"], 40)
)

# >10 PERSON
# 15 documenti casuali + 5 documenti difficili distinti

random_documents = random.sample(
    strata[">10"],
    15
)

# Ordiniamo i documenti per numero di PERSON decrescente
hard_documents = sorted(
    strata[">10"],
    key=lambda x: x["person_count"],
    reverse=True
)

# Selezioniamo i documenti difficili che NON sono
# già presenti nel campione casuale
random_ids = {
    document["document_id"]
    for document in random_documents
}

hard_documents = [
    document
    for document in hard_documents
    if document["document_id"] not in random_ids
]

sample.extend(random_documents)
sample.extend(hard_documents[:5])


# ============================================================
# CONTROLLO DUPLICATI
# ============================================================

unique_ids = {
    document["document_id"]
    for document in sample
}

if len(unique_ids) != 100:
    raise ValueError(
        f"Il campione contiene {len(unique_ids)} "
        f"documenti distinti invece di 100."
    )


# ============================================================
# ORDINE FINALE
# ============================================================

sample = sorted(
    sample,
    key=lambda x: x["document_id"]
)


# ============================================================
# CREAZIONE CARTELLA OUTPUT
# ============================================================

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SALVATAGGIO CSV
# ============================================================

output_file = (
    OUTPUT_PATH
    / "sample_100.csv"
)

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "document_id",
            "path",
            "split",
            "person_count",
            "length",
            "stratum"
        ]
    )

    writer.writeheader()
    writer.writerows(sample)


# ============================================================
# RISULTATI
# ============================================================

print("\n--- CAMPIONE CREATO ---")

print(f"Documenti selezionati: {len(sample)}")

print("\nDistribuzione del campione:")

for stratum in strata:

    count = sum(
        1
        for document in sample
        if document["stratum"] == stratum
    )

    print(
        f"{stratum}: {count}"
    )

print(
    f"\nFile salvato in:\n{output_file}"
)

print(
    f"\nSeed utilizzato: {SEED}"
)