import json
import os
import time
from pathlib import Path

from openai import OpenAI


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "dataset_100.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "openai_results_100.jsonl"

MODEL_NAME = "gpt-6-luna"
N_DOCUMENTS = 100


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
You are an anonymization system for email documents.

Your task is to identify and anonymize ALL PERSON NAMES in the email.

Instructions:

- Identify every occurrence of a person's name.
- Replace each person's name with a placeholder in the form PERSON_1,
  PERSON_2, etc.
- Use the same placeholder for repeated occurrences of the same person
  within the document.
- Do NOT anonymize organizations, companies, locations, email addresses,
  job titles, dates, or other non-person information.
- Do NOT rewrite, summarize, correct, reorder, or otherwise modify the email.
- Preserve the original text exactly except for substitutions of person names.
- Preserve line breaks, punctuation, headers, and formatting as much as possible.

Return the anonymized email and the detected entities.

EMAIL:
"""


# ============================================================
# JSON SCHEMA
# ============================================================

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
                    "replacement": {"type": "string"}
                },
                "required": ["original", "replacement"],
                "additionalProperties": False
            }
        }
    },
    "required": ["text_anonymized", "entities_detected"],
    "additionalProperties": False
}


# ============================================================
# CLIENT
# ============================================================

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ============================================================
# LETTURA DATASET
# ============================================================

documents = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            documents.append(json.loads(line))

documents = documents[:N_DOCUMENTS]


# ============================================================
# ELABORAZIONE
# ============================================================

results = []

print("=" * 60)
print("TEST OPENAI - M4 LLM SUBSTITUTION")
print("=" * 60)
print(f"Modello: {MODEL_NAME}")
print(f"Documenti: {len(documents)}")
print()

for i, document in enumerate(documents, start=1):

    document_id = document["document_id"]
    text_original = document["text_original"]

    print(f"[{i}/{len(documents)}] {document_id}")

    start_time = time.perf_counter()

    try:

        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": PROMPT
                },
                {
                    "role": "user",
                    "content": text_original
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "anonymization_result",
                    "strict": True,
                    "schema": OUTPUT_SCHEMA
                }
            }
        )

        elapsed = time.perf_counter() - start_time

        result = json.loads(response.output_text)

        usage = response.usage

        record = {
            "document_id": document_id,
            "model": MODEL_NAME,
            "text_original": text_original,
            "text_anonymized": result["text_anonymized"],
            "entities_detected": result["entities_detected"],
            "ground_truth": document["entities"],
            "elapsed_seconds": elapsed,
            "usage": {
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None
            },
            "status": "success"
        }

        results.append(record)

        print(f"    OK")
        print(f"    Tempo: {elapsed:.2f} s")
        print(f"    Entità: {len(result['entities_detected'])}")
        print()

    except Exception as e:

        elapsed = time.perf_counter() - start_time

        record = {
            "document_id": document_id,
            "model": MODEL_NAME,
            "status": "error",
            "error": str(e),
            "elapsed_seconds": elapsed
        }

        results.append(record)

        print(f"    ERRORE dopo {elapsed:.2f} s")
        print(f"    {type(e).__name__}: {e}")
        print()


# ============================================================
# SALVATAGGIO
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in results:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


print("=" * 60)
print(f"Risultati salvati in:")
print(OUTPUT_FILE)
print("=" * 60)