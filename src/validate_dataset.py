from pathlib import Path
import json


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_PATH
    / "data"
    / "processed"
    / "dataset_100.jsonl"
)


# ============================================================
# LETTURA E VALIDAZIONE
# ============================================================

total_documents = 0
total_entities = 0
errors = []

with open(
    DATASET_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line_number, line in enumerate(f, start=1):

        record = json.loads(line)

        total_documents += 1

        text = record["text_original"]

        for entity_index, entity in enumerate(
            record["entities"],
            start=1
        ):

            total_entities += 1

            start = entity["start"]
            end = entity["end"]

            extracted_text = text[start:end]

            expected_text = entity["text"]

            if extracted_text != expected_text:

                errors.append({
                    "line": line_number,
                    "document_id": record["document_id"],
                    "entity_index": entity_index,
                    "expected": expected_text,
                    "found": extracted_text,
                    "start": start,
                    "end": end
                })


# ============================================================
# RISULTATO
# ============================================================

print("\n--- VALIDAZIONE DATASET ---")

print(
    f"Documenti verificati: {total_documents}"
)

print(
    f"Entità verificate: {total_entities}"
)

print(
    f"Errori negli offset: {len(errors)}"
)


if errors:

    print("\n--- PRIMI ERRORI ---")

    for error in errors[:10]:

        print(
            f"\nDocumento: {error['document_id']}"
        )

        print(
            f"Entità: {error['entity_index']}"
        )

        print(
            f"Atteso: {error['expected']!r}"
        )

        print(
            f"Trovato: {error['found']!r}"
        )

        print(
            f"Offset: "
            f"{error['start']}:{error['end']}"
        )

else:

    print(
        "\n✓ Tutti gli offset sono corretti."
    )

    print(
        "✓ Il dataset può essere utilizzato "
        "come ground truth."
    )
    