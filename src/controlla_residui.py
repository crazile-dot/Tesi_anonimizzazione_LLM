
from pathlib import Path
import json
import csv
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent

ORIGINALE = PROJECT_DIR / "data" / "processed" / "corpus_test.jsonl"
ANONIMO = PROJECT_DIR / "data" / "processed" / "corpus_test_anonimo.jsonl"
RANKING = PROJECT_DIR / "data" / "processed" / "ranking_tfidf.csv"

NOMI = {
    "scott.neal@enron.com": "Scott Neal",
    "errol.mclaughlin@enron.com": "Errol McLaughlin",
    "drew.fossum@enron.com": "Drew Fossum",
    "mike.mcconnell@enron.com": "Mike McConnell",
}


def leggi_jsonl(percorso):
    with percorso.open("r", encoding="utf-8") as f:
        return {
            r["document_id"]: r
            for r in (json.loads(line) for line in f if line.strip())
        }


originali = leggi_jsonl(ORIGINALE)
anonimi = leggi_jsonl(ANONIMO)

with RANKING.open("r", encoding="utf-8-sig", newline="") as f:
    ranking = list(csv.DictReader(f))

print(f"Documenti controllati: {len(ranking)}\n")

for riga in ranking:
    doc_id = riga["document_id"]
    identita = riga["identita_vera"]

    email = next(
        (e for e, nome in NOMI.items()
         if nome.casefold() == identita.casefold()),
        None,
    )

    if email is None:
        print(f"Identità non riconosciuta: {identita}")
        continue

    nome = NOMI[email]

    testo_originale = originali[doc_id]["testo_originale"]
    testo_anonimo = anonimi[doc_id]["testo_anonimizzato"]

    # Ricerca del nome completo, ignorando maiuscole/minuscole
    pattern = re.compile(
        r"(?<!\w)" + re.escape(nome) + r"(?!\w)",
        re.IGNORECASE,
    )

    occorrenze_originali = len(pattern.findall(testo_originale))
    occorrenze_anonime = len(pattern.findall(testo_anonimo))

    print(f"Documento: {doc_id}")
    print(f"  Autore atteso: {nome}")
    print(f"  Nome nel testo originale: {occorrenze_originali}")
    print(f"  Nome nel testo anonimizzato: {occorrenze_anonime}")
    print()
