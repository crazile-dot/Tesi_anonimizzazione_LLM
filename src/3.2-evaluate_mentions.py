import json
from pathlib import Path


INPUT_FILE = Path("data/processed/dataset_100_baseline.jsonl")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_method(records, method_name):
    total_mentions = 0
    protected_mentions = 0
    missed_mentions = 0

    for record in records:
        text_original = record["text_original"]
        entities = record["entities"]
        transformed_text = record[method_name]

        # Ordiniamo le entità per posizione
        entities = sorted(entities, key=lambda e: e["start"])

        total_mentions += len(entities)

        # Verifica ogni entità annotata
        for entity in entities:
            start = entity["start"]
            end = entity["end"]

            original_value = text_original[start:end]

            # M1: la stringa originale non deve più comparire
            if method_name == "M1_redaction":
                replacement = "[REDACTED]"

            # M2: ricaviamo il placeholder dalla mappa
            elif method_name == "M2_pseudonymization":
                replacement = None

                for mapping in record["M2_anonymization_map"]:
                    if mapping["original"] == original_value:
                        replacement = mapping["replacement"]
                        break

                if replacement is None:
                    raise ValueError(
                        f"Mappatura M2 non trovata per: {original_value}"
                    )

            # M3: ricaviamo il placeholder dalla mappa
            elif method_name == "M3_hmac":
                replacement = None

                for mapping in record["M3_anonymization_map"]:
                    if mapping["original"] == original_value:
                        replacement = mapping["replacement"]
                        break

                if replacement is None:
                    raise ValueError(
                        f"Mappatura M3 non trovata per: {original_value}"
                    )

            else:
                raise ValueError(f"Metodo non supportato: {method_name}")

            # Una mention è considerata protetta se il replacement
            # compare nel testo trasformato.
            if replacement in transformed_text:
                protected_mentions += 1
            else:
                missed_mentions += 1

    precision = (
        protected_mentions / total_mentions
        if total_mentions > 0
        else None
    )

    recall = precision

    f1 = precision

    return {
        "method": method_name,
        "total_mentions": total_mentions,
        "protected_mentions": protected_mentions,
        "missed_mentions": missed_mentions,
        "mention_precision": precision,
        "mention_recall": recall,
        "mention_f1": f1,
    }


def main():
    records = load_jsonl(INPUT_FILE)

    methods = [
        "M1_redaction",
        "M2_pseudonymization",
        "M3_hmac",
    ]

    print(f"Documenti analizzati: {len(records)}")
    print()

    for method in methods:
        result = evaluate_method(records, method)

        print(method)
        print(f"  Mention totali:       {result['total_mentions']}")
        print(f"  Mention protette:     {result['protected_mentions']}")
        print(f"  Mention non protette: {result['missed_mentions']}")
        print(f"  Mention Recall:       {result['mention_recall']:.4f}")
        print()


if __name__ == "__main__":
    main()