
from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT = PROJECT_DIR / "data" / "processed" / "ground_truth_test.jsonl"

documenti = 0
identificativi = set()
menzioni = 0
documenti_senza_nomi = 0

with INPUT.open("r", encoding="utf-8") as f:
    for linea in f:
        doc = json.loads(linea)
        documenti += 1

        identificativi.add(doc["document_id"])
        menzioni += len(doc["nomi_annotati"])

        if not doc["nomi_annotati"]:
            documenti_senza_nomi += 1

print(f"Documenti letti: {documenti}")
print(f"Identificativi univoci: {len(identificativi)}")
print(f"Menzioni annotate: {menzioni}")
print(f"Documenti senza nomi annotati: {documenti_senza_nomi}")
print(f"Identificativi duplicati: {documenti - len(identificativi)}")
