import json
from pathlib import Path

path = Path("data/processed/openai_results_100.jsonl")

with path.open("r", encoding="utf-8") as f:
    record = json.loads(next(f))

print("DOCUMENTO:", record["document_id"])
print("TESTO ORIGINALE (inizio):")
print(repr(record["text_original"][:700]))

print("\nTESTO ANONIMIZZATO (inizio):")
print(repr(record["text_anonymized"][:700]))

print("\nGROUND TRUTH:")
print(json.dumps(record["ground_truth"], ensure_ascii=False, indent=2))

print("\nENTITIES DETECTED:")
print(json.dumps(record["entities_detected"], ensure_ascii=False, indent=2))