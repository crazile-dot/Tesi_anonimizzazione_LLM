
from pathlib import Path
import json
import re
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT = PROJECT_DIR / "data" / "processed" / "corpus_test.jsonl"
ANONIMO = PROJECT_DIR / "data" / "processed" / "corpus_test_anonimo.jsonl"

def normalizza(nome):
    return " ".join(nome.split()).casefold()

documenti_originali = {}

with INPUT.open("r", encoding="utf-8") as f:
    for linea in f:
        doc = json.loads(linea)
        documenti_originali[doc["document_id"]] = doc

casi = []
totale_documenti = 0

with ANONIMO.open("r", encoding="utf-8") as f:
    for linea in f:
        anon = json.loads(linea)
        originale = documenti_originali[anon["document_id"]]
        testo = anon["testo_anonimizzato"]

        # Conta i segnaposto persona distinti nel documento.
        segnaposto = set(
            re.findall(r"\[PERSONA_\d+\]", testo)
        )

        # Conta le varianti annotate distinte.
        nomi = {
            normalizza(n)
            for n in originale["nomi_annotati"]
            if normalizza(n)
        }

        totale_documenti += 1

        if len(segnaposto) > len(nomi):
            casi.append({
                "document_id": anon["document_id"],
                "varianti_annotate": len(nomi),
                "segnaposto_distinti": len(segnaposto)
            })

print(f"Documenti controllati: {totale_documenti}")
print(f"Documenti con più segnaposto che varianti annotate: {len(casi)}")

for caso in casi[:10]:
    print(caso)
