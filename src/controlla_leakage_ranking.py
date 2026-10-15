
from pathlib import Path
import json
import csv
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
TEST_ANONIMO = PROJECT_DIR / "data" / "processed" / "corpus_test_anonimo.jsonl"
RANKING = PROJECT_DIR / "data" / "processed" / "ranking_tfidf.csv"

EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

with TEST_ANONIMO.open("r", encoding="utf-8") as f:
    documenti = {
        r["document_id"]: r
        for r in (json.loads(line) for line in f if line.strip())
    }

with RANKING.open("r", encoding="utf-8-sig", newline="") as f:
    ranking = list(csv.DictReader(f))

print(f"Email valutate: {len(ranking)}")
print()

totale_email_residue = 0

for riga in ranking:
    doc_id = riga["document_id"]
    doc = documenti.get(doc_id)

    if doc is None:
        print(f"DOCUMENTO NON TROVATO: {doc_id}")
        continue

    testo = doc["testo_anonimizzato"]
    email = sorted(set(EMAIL_RE.findall(testo)))

    totale_email_residue += len(email)

    print(f"Documento: {doc_id}")
    print(f"  Mittente atteso: {riga['identita_vera']}")
    print(f"  Indirizzi email residui: {len(email)}")

    if email:
        print(f"  Esempi: {email[:5]}")

print()
print(f"Totale indirizzi email residui: {totale_email_residue}")
