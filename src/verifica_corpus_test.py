
from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_DIR / "data" / "processed" / "corpus_test.jsonl"

totale = 0
con_annotazioni = 0
annotazioni_totali = 0
testi_vuoti = 0
esempi = []

with CORPUS.open("r", encoding="utf-8") as f:
    for linea in f:
        documento = json.loads(linea)
        totale += 1

        nomi = documento["nomi_annotati"]
        annotazioni_totali += len(nomi)

        if nomi:
            con_annotazioni += 1

        if not documento["testo_originale"].strip():
            testi_vuoti += 1

        if len(esempi) < 3 and nomi:
            esempi.append(documento)

print(f"Documenti: {totale}")
print(f"Documenti con annotazioni: {con_annotazioni}")
print(f"Annotazioni totali: {annotazioni_totali}")
print(f"Testi originali vuoti: {testi_vuoti}")

print("\n--- ESEMPI ---")
for documento in esempi:
    print(f"\nID: {documento['document_id']}")
    print(f"Annotazioni: {documento['numero_annotazioni']}")
    print(f"Nomi estratti: {documento['nomi_annotati'][:10]}")
    print(f"Header presente: {bool(documento['header_originale'].strip())}")
    print(f"Corpo presente: {bool(documento['corpo_originale'].strip())}")
