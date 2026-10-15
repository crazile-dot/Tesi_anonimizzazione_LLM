
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "dataset_680_baseline.jsonl"

# Nuovo file: non riutilizza i risultati della vecchia M4.
OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" / "openai_results_680_m4_v2.jsonl"
)

MODEL_NAME = "gpt-6-luna"

PROMPT = """
You are an anonymization system for email documents.

Your task is to anonymize PERSON NAMES, EMAIL ADDRESSES,
and MESSAGE-ID values throughout the entire email document.

PERSON NAMES:
- Identify every occurrence of a person's name.
- Replace each person with PERSON_1, PERSON_2, etc.
- Use the same placeholder for repeated occurrences of the same person
  within the document.

EMAIL ADDRESSES:
- Anonymize every email address, including addresses in From, To, Cc,
  Bcc, Reply-To, signatures, quoted text, and the message body.
- Replace each distinct email address with EMAIL_1, EMAIL_2, etc.
- Reuse the same placeholder for repeated occurrences of the same address.
- Do not leave any original email address visible.

MESSAGE-ID:
- Replace the value of every Message-ID header with [MESSAGE_ID].
- Do not preserve the original Message-ID value.

OTHER INFORMATION:
- Do not anonymize organizations, companies, locations, job titles,
  dates, or other non-person information unless it is part of an
  email address or Message-ID value.
- Do not rewrite, summarize, correct, reorder, or otherwise modify
  the email.
- Preserve the original content, line breaks, punctuation, and formatting
  as much as possible, except for the required substitutions.

Return the anonymized email and the detected entities.
Each detected entity must contain its original value and replacement.

EMAIL:
"""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "text_anonymized": {"type": "string"},
        "entities_detected": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original": {"type": "string"},
                    "replacement": {"type": "string"},
                },
                "required": ["original", "replacement"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["text_anonymized", "entities_detected"],
    "additionalProperties": False,
}


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ============================================================
# CONTROLLI DETERMINISTICI SU EMAIL E MESSAGE-ID
# ============================================================

EMAIL_RE = re.compile(
    r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
)

MESSAGE_ID_RE = re.compile(
    r"(?im)^(Message-ID\s*:\s*).*$"
)


def anonimizza_email_e_message_id(testo):
    """
    Garantisce la sostituzione degli indirizzi email e del valore
    dell'intestazione Message-ID anche se il modello li ha lasciati.
    La mappatura degli indirizzi è locale al singolo documento.
    """

    mappatura_email = {}

    def sostituisci_email(match):
        email_originale = match.group(0)
        chiave = email_originale.casefold()

        if chiave not in mappatura_email:
            mappatura_email[chiave] = (
                f"EMAIL_{len(mappatura_email) + 1}"
            )

        return mappatura_email[chiave]

    testo = EMAIL_RE.sub(sostituisci_email, testo)

    testo = MESSAGE_ID_RE.sub(
        lambda match: match.group(1) + "[MESSAGE_ID]",
        testo,
    )

    entita_email = [
        {
            "original": email,
            "replacement": placeholder,
        }
        for email, placeholder in [
            (email, placeholder)
            for email, placeholder in []
        ]
    ]

    # Ricostruzione delle entità email dalla mappatura.
    # La chiave normalizzata viene usata per mantenere la coerenza.
    entita_email = [
        {
            "original": email,
            "replacement": placeholder,
        }
        for email, placeholder in mappatura_email.items()
    ]

    return testo, entita_email


# ============================================================
# LETTURA DATASET
# ============================================================

documents = []

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            documents.append(json.loads(line))

print("=" * 60)
print("ANONIMIZZAZIONE M4 V2 - 680 DOCUMENTI")
print("=" * 60)
print(f"Modello: {MODEL_NAME}")
print(f"Documenti in input: {len(documents)}")


# ============================================================
# RECUPERO RISULTATI DELLA NUOVA VERSIONE
# ============================================================

risultati_per_id = {}

if OUTPUT_FILE.exists():
    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if (
                record.get("status") == "success"
                and record.get("document_id")
                and isinstance(record.get("text_anonymized"), str)
                and isinstance(record.get("entities_detected"), list)
            ):
                risultati_per_id[record["document_id"]] = record

print(f"Risultati già completati in M4 V2: {len(risultati_per_id)}")
print()


# ============================================================
# ELABORAZIONE CON SALVATAGGIO PROGRESSIVO
# ============================================================

for i, document in enumerate(documents, start=1):
    document_id = document["document_id"]

    if document_id == "donohoe-t_all_documents_79":
        print(
            f"[{i}/{len(documents)}] {document_id} — "
            "saltato per timeout nella precedente elaborazione."
        )

        record = {
            "document_id": document_id,
            "model": MODEL_NAME,
            "version": "M4_v2",
            "status": "error",
            "error": (
                "Email esclusa dalla nuova elaborazione: "
                "timeout prolungato nella precedente esecuzione."
            ),
        }

        with OUTPUT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        continue


    if document_id in risultati_per_id:
        print(
            f"[{i}/{len(documents)}] {document_id} — già completato, salto."
        )
        continue

    text_original = document["testo_originale"]

    print(f"[{i}/{len(documents)}] {document_id}")

    start_time = time.perf_counter()

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": text_original},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "anonymization_result",
                    "strict": True,
                    "schema": OUTPUT_SCHEMA,
                }
            },
        )

        elapsed = time.perf_counter() - start_time
        result = json.loads(response.output_text)
        usage = response.usage

        # Controllo deterministico aggiuntivo.
        testo_m4, entita_email = anonimizza_email_e_message_id(
            result["text_anonymized"]
        )

        entita_rilevate = (
            result["entities_detected"] + entita_email
        )

        record = {
            "document_id": document_id,
            "model": MODEL_NAME,
            "version": "M4_v2",
            "text_original": text_original,
            "text_anonymized": testo_m4,
            "entities_detected": entita_rilevate,
            "elapsed_seconds": elapsed,
            "usage": {
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
            },
            "status": "success",
        }

        # Controlli minimi prima del salvataggio.
        email_residue = EMAIL_RE.findall(testo_m4)
        if email_residue:
            raise ValueError(
                f"Email residue dopo l'anonimizzazione: {email_residue[:3]}"
            )

        if re.search(
            r"(?im)^Message-ID\s*:\s*(?!\[MESSAGE_ID\])\S+",
            testo_m4,
        ):
            raise ValueError("Message-ID originale ancora presente.")

        print(f"    OK — {elapsed:.2f} s")
        print(f"    Entità rilevate dal modello: "
              f"{len(result['entities_detected'])}")
        print(f"    Email anonimizzate dal controllo: "
              f"{len(entita_email)}")

    except Exception as e:
        elapsed = time.perf_counter() - start_time

        record = {
            "document_id": document_id,
            "model": MODEL_NAME,
            "version": "M4_v2",
            "status": "error",
            "error": str(e),
            "elapsed_seconds": elapsed,
        }

        print(f"    ERRORE — {type(e).__name__}: {e}")

    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if record["status"] == "success":
        risultati_per_id[document_id] = record

    print()


# ============================================================
# RIEPILOGO
# ============================================================

successi = sum(
    1
    for record in risultati_per_id.values()
    if record.get("status") == "success"
)

errori = len(documents) - successi

print("=" * 60)
print("ELABORAZIONE M4 V2 TERMINATA")
print(f"Documenti nel dataset: {len(documents)}")
print(f"Documenti completati: {successi}")
print(f"Documenti mancanti o in errore: {errori}")
print(f"Risultati salvati in: {OUTPUT_FILE}")
print("=" * 60)
