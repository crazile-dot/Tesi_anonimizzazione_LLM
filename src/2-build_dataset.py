'''
build_dataset.py

Scopo:
Costruire un dataset strutturato a partire dal campione di 100 email
selezionate in precedenza, preservando il testo originale dei nomi
annotati e registrando le relative entità PERSON.

Funzionamento:
1. Legge l'elenco dei documenti da data/processed/sample_100.csv.
2. Carica il contenuto di ciascuna email originale.
3. Individua le annotazioni racchiuse nei tag <true_name>...</true_name>.
4. Rimuove i tag dal testo, mantenendo i nomi delle persone.
5. Registra per ogni entità il tipo, il testo e gli offset start/end
   rispetto al testo ottenuto dopo la rimozione dei tag.
6. Salva un record JSON per ogni documento in
   data/processed/dataset_100.jsonl.
7. Stampa un riepilogo del numero di documenti e delle entità PERSON
   estratte.

Input:
- data/processed/sample_100.csv
- File delle email originali indicati nel CSV.

Output:
- data/processed/dataset_100.jsonl

Nota:
Questo script prepara il dataset annotato per le successive fasi
sperimentali. Non esegue alcuna anonimizzazione: i nomi delle persone
rimangono presenti nel testo.
'''

from pathlib import Path
import json
import csv
import re


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parent.parent

SAMPLE_FILE = (
    PROJECT_PATH
    / "data"
    / "processed"
    / "sample_100.csv"
)

OUTPUT_FILE = (
    PROJECT_PATH
    / "data"
    / "processed"
    / "dataset_100.jsonl"
)


# ============================================================
# PATTERN DELLE ANNOTAZIONI
# ============================================================

PERSON_PATTERN = re.compile(
    r"<true_name>(.*?)</true_name>",
    flags=re.DOTALL
)


# ============================================================
# FUNZIONE DI PULIZIA E ANNOTAZIONE
# ============================================================

def clean_text_and_extract_entities(text):

    entities = []

    # Costruiamo il testo pulito pezzo per pezzo
    clean_parts = []

    # Posizione corrente nel testo originale
    original_position = 0

    # Posizione corrente nel testo pulito
    clean_position = 0

    # Troviamo tutte le annotazioni
    matches = PERSON_PATTERN.finditer(text)

    for match in matches:

        # ----------------------------------------------------
        # Parte di testo prima del tag <true_name>
        # ----------------------------------------------------

        before = text[
            original_position:match.start()
        ]

        clean_parts.append(before)

        clean_position += len(before)

        # ----------------------------------------------------
        # Contenuto dell'annotazione
        # ----------------------------------------------------

        raw_entity_text = match.group(1)

        entity_text = raw_entity_text.strip()

        # Posizione del nome senza gli spazi iniziali
        leading_whitespace = (
            len(raw_entity_text)
            - len(raw_entity_text.lstrip())
        )

        # Offset nel testo PULITO
        start = clean_position + leading_whitespace

        end = start + len(entity_text)

        entities.append({
            "type": "PERSON",
            "text": entity_text,
            "start": start,
            "end": end
        })

        # Aggiungiamo il nome senza i tag
        clean_parts.append(raw_entity_text)

        clean_position += len(raw_entity_text)

        # Aggiorniamo la posizione nel testo originale
        original_position = match.end()

    # --------------------------------------------------------
    # Parte finale del documento
    # --------------------------------------------------------

    remaining = text[original_position:]

    clean_parts.append(remaining)

    # --------------------------------------------------------
    # Testo finale
    # --------------------------------------------------------

    clean_text = "".join(clean_parts)

    return clean_text, entities


# ============================================================
# LETTURA DEL SAMPLE
# ============================================================

with open(
    SAMPLE_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)
    documents = list(reader)


print(
    f"Documenti da elaborare: {len(documents)}"
)


# ============================================================
# CREAZIONE DATASET
# ============================================================

dataset = []

for i, document in enumerate(
    documents,
    start=1
):

    file_path = Path(document["path"])

    # --------------------------------------------------------
    # Lettura documento originale
    # --------------------------------------------------------

    text_raw = file_path.read_text(
        encoding="latin-1",
        errors="replace"
    )

    # --------------------------------------------------------
    # Rimozione tag + estrazione PERSON
    # --------------------------------------------------------

    text_clean, entities = (
        clean_text_and_extract_entities(
            text_raw
        )
    )

    # --------------------------------------------------------
    # Record
    # --------------------------------------------------------

    record = {
        "document_id": document["document_id"],
        "text_original": text_clean,
        "entities": entities
    }

    dataset.append(record)

    print(
        f"[{i:3}/{len(documents)}] "
        f"{document['document_id']} "
        f"→ {len(entities)} PERSON"
    )


# ============================================================
# SALVATAGGIO JSONL
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for record in dataset:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# CONTROLLO FINALE
# ============================================================

total_entities = sum(
    len(d["entities"])
    for d in dataset
)

print("\n--- DATASET CREATO ---")

print(
    f"Record creati: {len(dataset)}"
)

print(
    f"File salvato in:\n{OUTPUT_FILE}"
)

print(
    f"PERSON totali: {total_entities}"
)