
from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT = PROJECT_DIR / "data" / "processed" / "corpus_test.jsonl"
OUTPUT = PROJECT_DIR / "data" / "processed" / "ground_truth_test.jsonl"

documenti = 0
menzioni = 0

with INPUT.open("r", encoding="utf-8") as sorgente, \
     OUTPUT.open("w", encoding="utf-8", newline="\n") as destinazione:

    for linea in sorgente:
        doc = json.loads(linea)

        nomi = doc["nomi_annotati"]

        riferimento = {
            "document_id": doc["document_id"],
            "nomi_annotati": nomi,
            "numero_annotazioni": len(nomi)
        }

        destinazione.write(
            json.dumps(riferimento, ensure_ascii=False) + "\n"
        )

        documenti += 1
        menzioni += len(nomi)

print(f"Documenti nel riferimento: {documenti}")
print(f"Menzioni annotate nel riferimento: {menzioni}")
print(f"File creato: {OUTPUT}")
