'''
Lo script analizza tutti i documenti del dataset EnronRandom-XML per caratterizzare 
la distribuzione delle entità PERSON e la lunghezza dei testi. Conta le occorrenze di 
PERSON individuate tramite i tag <true_name>, calcola le statistiche generali del corpus 
e suddivide i documenti in quattro fasce in base al numero di entità annotate (0, 1–2, 
3–10 e oltre 10). I risultati, stampati a terminale, consentono di descrivere il dataset 
originale e supportano la scelta dei criteri di campionamento. Lo script è esclusivamente 
analitico e non produce file né modifica i dati.
'''
from pathlib import Path
import re
from collections import Counter, defaultdict


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


# ============================================================
# RICERCA DEI DOCUMENTI
# ============================================================

files = [
    f
    for f in DATASET_PATH.rglob("*")
    if f.is_file() and f.suffix != ".zip"
]

print(f"Documenti totali: {len(files)}")


# ============================================================
# ANALISI
# ============================================================

pattern = r"<true_name>(.*?)</true_name>"

person_counts = Counter()
lengths_by_person = defaultdict(list)

total_person = 0
documents_with_person = 0
document_lengths = []


for file_path in files:

    text = file_path.read_text(
        encoding="latin-1",
        errors="replace"
    )

    length = len(text)

    matches = re.findall(
        pattern,
        text,
        flags=re.DOTALL
    )

    count = len(matches)

    # Statistiche generali
    person_counts[count] += 1
    total_person += count
    document_lengths.append(length)

    if count > 0:
        documents_with_person += 1

    # Lunghezza associata al numero di PERSON
    lengths_by_person[count].append(length)


# ============================================================
# TABELLA DETTAGLIATA
# ============================================================

print("\n--- PERSON E LUNGHEZZA DOCUMENTI ---")

print(
    f"{'PERSON':>8} | "
    f"{'DOCUMENTI':>10} | "
    f"{'LUNGHEZZA MEDIA':>16}"
)

print("-" * 42)

for count in sorted(person_counts):

    documents = person_counts[count]

    average_length = (
        sum(lengths_by_person[count])
        / documents
    )

    print(
        f"{count:>8} | "
        f"{documents:>10} | "
        f"{average_length:>16.0f}"
    )


# ============================================================
# QUATTRO FASCE
# ============================================================

groups = {
    "0": lambda x: x == 0,
    "1-2": lambda x: 1 <= x <= 2,
    "3-10": lambda x: 3 <= x <= 10,
    ">10": lambda x: x > 10
}

print("\n--- ANALISI PER FASCE ---")

print(
    f"{'FASCIA':>10} | "
    f"{'DOCUMENTI':>10} | "
    f"{'LUNGHEZZA MEDIA':>16}"
)

print("-" * 44)

for group_name, condition in groups.items():

    selected_counts = [
        count
        for count in person_counts
        if condition(count)
    ]

    num_documents = sum(
        person_counts[count]
        for count in selected_counts
    )

    selected_lengths = []

    for count in selected_counts:
        selected_lengths.extend(
            lengths_by_person[count]
        )

    average_length = (
        sum(selected_lengths)
        / len(selected_lengths)
        if selected_lengths
        else 0
    )

    print(
        f"{group_name:>10} | "
        f"{num_documents:>10} | "
        f"{average_length:>16.0f}"
    )


# ============================================================
# STATISTICHE GENERALI
# ============================================================

print("\n--- STATISTICHE GENERALI ---")

print(f"Documenti totali: {len(files)}")

print(
    f"Documenti con almeno una PERSON: "
    f"{documents_with_person}"
)

print(
    f"Documenti senza PERSON: "
    f"{len(files) - documents_with_person}"
)

print(f"PERSON totali: {total_person}")

print(
    f"Media PERSON per documento: "
    f"{total_person / len(files):.2f}"
)

print(
    f"Lunghezza media: "
    f"{sum(document_lengths) / len(document_lengths):.0f} caratteri"
)

print(
    f"Lunghezza minima: "
    f"{min(document_lengths)} caratteri"
)

print(
    f"Lunghezza massima: "
    f"{max(document_lengths)} caratteri"
)