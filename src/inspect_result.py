
import json
from pathlib import Path


# =========================================================
# CONFIGURAZIONE
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_llm_one.json"
)


# =========================================================
# LETTURA RISULTATO
# =========================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    result = json.load(f)


ground_truth = result["ground_truth"]
entities_detected = result["entities_detected"]


# =========================================================
# GROUND TRUTH
# =========================================================

print("=" * 60)
print("GROUND TRUTH")
print("=" * 60)

print(f"Numero entità: {len(ground_truth)}")
print()

for i, entity in enumerate(ground_truth, start=1):

    print(
        f"{i}. "
        f"type={entity['type']} | "
        f"text='{entity['text']}' | "
        f"start={entity['start']} | "
        f"end={entity['end']}"
    )


# =========================================================
# ENTITÀ RILEVATE DA GEMINI
# =========================================================

print()
print("=" * 60)
print("ENTITÀ RILEVATE DA GEMINI")
print("=" * 60)

print(f"Numero entità: {len(entities_detected)}")
print()

for i, entity in enumerate(entities_detected, start=1):

    print(
        f"{i}. "
        f"original='{entity['original']}' | "
        f"replacement='{entity['replacement']}'"
    )
