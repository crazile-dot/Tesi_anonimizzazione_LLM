'''
script di verifica della correttezza delle strategie di anonimizzazione M1, M2 e M3
'''

import json
import re
import hashlib
import hmac


INPUT_FILE = r"data\processed\dataset_100_baseline.jsonl"

HMAC_SECRET_KEY = b"tesi-anonimizzazione-secret-key"


def apply_replacements(text, replacements):
    """
    Applica le sostituzioni partendo dalla fine del testo,
    così gli offset originali rimangono validi.
    """

    result = text

    for start, end, replacement in sorted(
        replacements,
        key=lambda x: x[0],
        reverse=True
    ):
        result = result[:start] + replacement + result[end:]

    return result


def expected_m1(text, entities):
    """
    Costruisce il risultato atteso per M1.
    """

    replacements = []

    for entity in entities:
        replacements.append(
            (
                entity["start"],
                entity["end"],
                "[REDACTED]"
            )
        )

    return apply_replacements(text, replacements)


def expected_m2(text, entities):
    """
    Costruisce il risultato atteso per M2.

    La stessa stringa PERSON viene associata
    allo stesso placeholder.
    """

    mapping = {}
    counter = 1
    replacements = []

    for entity in entities:

        original = entity["text"]

        if original not in mapping:
            mapping[original] = f"PERSON_{counter}"
            counter += 1

        replacements.append(
            (
                entity["start"],
                entity["end"],
                mapping[original]
            )
        )

    return apply_replacements(text, replacements)


def hmac_person(name):
    """
    Calcola lo stesso identificativo HMAC utilizzato
    dallo script baseline.
    """

    digest = hmac.new(
        HMAC_SECRET_KEY,
        name.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()[:12]

    return f"PERSON_{digest}"


def expected_m3(text, entities):
    """
    Costruisce il risultato atteso per M3.
    """

    mapping = {}
    replacements = []

    for entity in entities:

        original = entity["text"]

        if original not in mapping:
            mapping[original] = hmac_person(original)

        replacements.append(
            (
                entity["start"],
                entity["end"],
                mapping[original]
            )
        )

    return apply_replacements(text, replacements)


def count_occurrences(text, substring):
    """
    Conta le occorrenze non sovrapposte.
    """

    return len(re.findall(
        re.escape(substring),
        text
    ))


def evaluate_document(record):
    """
    Valuta un singolo documento.
    """

    document_id = record["document_id"]
    original = record["text_original"]
    entities = record["entities"]

    actual_m1 = record["M1_redaction"]
    actual_m2 = record["M2_pseudonymization"]
    actual_m3 = record["M3_hmac"]

    expected_results = {
        "M1": expected_m1(original, entities),
        "M2": expected_m2(original, entities),
        "M3": expected_m3(original, entities)
    }

    actual_results = {
        "M1": actual_m1,
        "M2": actual_m2,
        "M3": actual_m3
    }

    evaluation = {
        "document_id": document_id,
        "num_gold_mentions": len(entities),
        "methods": {}
    }

    for method in ["M1", "M2", "M3"]:

        expected = expected_results[method]
        actual = actual_results[method]

        exact_match = expected == actual

        evaluation["methods"][method] = {
            "exact_match": exact_match,
            "expected_length": len(expected),
            "actual_length": len(actual),
            "length_difference": len(actual) - len(expected)
        }

    return evaluation


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    print("=" * 70)
    print("VALUTAZIONE POSITION-AWARE BASELINE")
    print("=" * 70)

    print(f"Documenti: {len(records)}")
    print()

    results = []

    for record in records:

        result = evaluate_document(record)
        results.append(result)

    methods = ["M1", "M2", "M3"]

    for method in methods:

        exact_correct = sum(
            1
            for result in results
            if result["methods"][method]["exact_match"]
        )

        total = len(results)

        print(f"{method}")
        print("-" * 40)
        print(
            f"Documenti corretti: "
            f"{exact_correct}/{total}"
        )

        print(
            f"Percentuale: "
            f"{exact_correct / total:.4f}"
        )

        if exact_correct < total:

            print("Documenti con differenze:")

            for result in results:

                if not result["methods"][method]["exact_match"]:

                    print(
                        f"  - {result['document_id']} "
                        f"(length diff: "
                        f"{result['methods'][method]['length_difference']})"
                    )

        print()

    # Controllo complessivo
    all_correct = all(
        result["methods"][method]["exact_match"]
        for result in results
        for method in methods
    )

    print("=" * 70)

    if all_correct:
        print("✓ TUTTI I RISULTATI M1-M3 SONO CORRETTI")
    else:
        print("⚠ SONO PRESENTI DIFFERENZE DA ANALIZZARE")

    print("=" * 70)


if __name__ == "__main__":
    main()