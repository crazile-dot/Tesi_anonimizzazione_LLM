
from pathlib import Path
import json
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT = PROJECT_DIR / "data" / "processed" / "corpus_test_anonimo.jsonl"

TAG_PATTERN = re.compile(
    r"</?true_name>",
    re.IGNORECASE
)
EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE
)

documenti = 0
tag_residui = 0
email_residue = 0
placeholder_persona = 0
placeholder_email = 0
esempi = []

with INPUT.open("r", encoding="utf-8") as f:
    for linea in f:
        doc = json.loads(linea)
        testo = doc["testo_anonimizzato"]
        documenti += 1

        tag_residui += len(TAG_PATTERN.findall(testo))
        email_residue += len(EMAIL_PATTERN.findall(testo))
        placeholder_persona += len(
            re.findall(r"\[PERSONA(?:_\d+)?\]", testo)
        )
        placeholder_email += len(
            re.findall(r"\[EMAIL_\d+\]", testo)
        )

        if len(esempi) < 2:
            esempi.append((doc["document_id"], testo[:1000]))

print(f"Documenti controllati: {documenti}")
print(f"Tag true_name residui: {tag_residui}")
print(f"Indirizzi email residui: {email_residue}")
print(f"Segnaposto persona: {placeholder_persona}")
print(f"Segnaposto email: {placeholder_email}")

print("\n--- ESEMPI ANONIMIZZATI ---")
for doc_id, testo in esempi:
    print(f"\n{doc_id}\n{testo}\n")
