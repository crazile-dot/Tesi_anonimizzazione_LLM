"""
anonymize_baseline.py

Scopo:
Creare le condizioni di riferimento e le baseline di anonimizzazione
da utilizzare negli esperimenti comparativi sul corpus di 100 documenti.

Funzionamento:
1. Legge il dataset annotato dataset_100.jsonl, contenente i testi
   originali e le entità PERSON con i relativi offset.
2. Mantiene il testo originale come condizione di riferimento M0.
3. Applica M1 (redaction), sostituendo ogni nome annotato con
   il placeholder [REDACTED].
4. Applica M2 (pseudonimizzazione deterministica), sostituendo
   i nomi con identificativi PERSON_N e mantenendo la coerenza
   delle sostituzioni per le occorrenze dello stesso nome
   all'interno del documento.
5. Applica M3 (pseudonimizzazione mediante HMAC-SHA256),
   generando identificativi deterministici a partire dai nomi
   e da una chiave segreta.
6. Registra i testi trasformati e le mappe di sostituzione
   per M2 e M3.
7. Salva tutti i risultati in dataset_100_baseline.jsonl,
   un record JSON per documento.

Input:
- data/processed/dataset_100.jsonl

Output:
- data/processed/dataset_100_baseline.jsonl

Nota:
Lo script applica le trasformazioni esclusivamente alle entità
PERSON annotate nel dataset di ingresso. Non esegue rilevamento
automatico di nuovi nomi né anonimizzazione mediante LLM.
"""

from pathlib import Path
import json
import hashlib
import hmac


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_PATH
    / "data"
    / "processed"
    / "dataset_100.jsonl"
)

OUTPUT_FILE = (
    PROJECT_PATH
    / "data"
    / "processed"
    / "dataset_100_baseline.jsonl"
)

# Chiave segreta utilizzata per M3
# NON deve essere salvata nel dataset di output.
HMAC_SECRET_KEY = b"tesi-anonimizzazione-secret-key"

TAG_PATTERN = re.compile(
    r"<true_name>(.*?)</true_name>",
    re.IGNORECASE | re.DOTALL
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE
)

MESSAGE_ID_PATTERN = re.compile(
    r"(?im)^Message-ID:.*$"
)


def normalizza_nome(nome):
    return " ".join(nome.split()).casefold()


# ============================================================
# FUNZIONE DI SOSTITUZIONE
# ============================================================

def apply_replacements(text, replacements):

    anonymized_text = text

    # Sostituiamo dalla fine verso l'inizio
    # per mantenere validi gli offset.
    for replacement in reversed(replacements):

        start = replacement["start"]
        end = replacement["end"]
        new_text = replacement["replacement"]

        anonymized_text = (
            anonymized_text[:start]
            + new_text
            + anonymized_text[end:]
        )

    return anonymized_text


# ============================================================
# M1 - REDACTION
# ============================================================

def create_m1(text, entities):

    replacements = []

    for entity in entities:

        replacements.append({
            "start": entity["start"],
            "end": entity["end"],
            "replacement": "[REDACTED]"
        })

    return apply_replacements(
        text,
        replacements
    )


# ============================================================
# M2 - PSEUDONIMIZZAZIONE DETERMINISTICA
# ============================================================

def create_m2(text, entities):

    entities_sorted = sorted(
        entities,
        key=lambda e: e["start"]
    )

    replacements = []
    anonymization_map = []

    person_counter = 0

    name_to_replacement = {}

    for entity in entities_sorted:

        original = entity["text"]

        if original not in name_to_replacement:

            person_counter += 1

            replacement = (
                f"PERSON_{person_counter}"
            )

            name_to_replacement[
                original
            ] = replacement

            anonymization_map.append({
                "original": original,
                "replacement": replacement
            })

        else:

            replacement = name_to_replacement[
                original
            ]

        replacements.append({
            "start": entity["start"],
            "end": entity["end"],
            "replacement": replacement
        })

    anonymized_text = apply_replacements(
        text,
        replacements
    )

    return anonymized_text, anonymization_map


# ============================================================
# M3 - PSEUDONIMIZZAZIONE MEDIANTE HMAC
# ============================================================

def create_m3(text, entities):

    replacements = []
    anonymization_map = []

    name_to_replacement = {}

    for entity in entities:

        original = entity["text"]

        if original not in name_to_replacement:

            digest = hmac.new(
                HMAC_SECRET_KEY,
                original.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()

            replacement = (
                f"PERSON_{digest[:12]}"
            )

            name_to_replacement[
                original
            ] = replacement

            anonymization_map.append({
                "original": original,
                "replacement": replacement
            })

        else:

            replacement = name_to_replacement[
                original
            ]

        replacements.append({
            "start": entity["start"],
            "end": entity["end"],
            "replacement": replacement
        })

    anonymized_text = apply_replacements(
        text,
        replacements
    )

    return anonymized_text, anonymization_map


# ============================================================
# ELABORAZIONE DI UN DOCUMENTO
# ============================================================

def anonymize_document(record):

    text = record["text_original"]
    entities = record["entities"]

    # --------------------------------------------------------
    # M0 - DOCUMENTO ORIGINALE
    # --------------------------------------------------------

    text_m0 = text

    # --------------------------------------------------------
    # M1 - REDACTION
    # --------------------------------------------------------

    text_m1 = create_m1(
        text,
        entities
    )

    # --------------------------------------------------------
    # M2 - PSEUDONIMIZZAZIONE
    # --------------------------------------------------------

    text_m2, map_m2 = create_m2(
        text,
        entities
    )

    # --------------------------------------------------------
    # M3 - HMAC
    # --------------------------------------------------------

    text_m3, map_m3 = create_m3(
        text,
        entities
    )

    # --------------------------------------------------------
    # RECORD FINALE
    # --------------------------------------------------------

    return {
        "document_id": record["document_id"],
        "text_original": text,
        "entities": entities,

        "M0_original": text_m0,

        "M1_redaction": text_m1,

        "M2_pseudonymization": text_m2,
        "M2_anonymization_map": map_m2,

        "M3_hmac": text_m3,
        "M3_anonymization_map": map_m3
    }


# ============================================================
# MAIN
# ============================================================

def main():

    records_processed = 0
    total_entities = 0

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as infile, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line in infile:

            record = json.loads(line)

            result = anonymize_document(
                record
            )

            outfile.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                + "\n"
            )

            records_processed += 1
            total_entities += len(
                record["entities"]
            )

    print("\n--- BASELINE CREATA ---")

    print(
        f"Documenti elaborati: {records_processed}"
    )

    print(
        f"Entità elaborate: {total_entities}"
    )

    print("\nFile salvato in:")

    print(OUTPUT_FILE)


# ============================================================
# AVVIO
# ============================================================

if __name__ == "__main__":
    main()