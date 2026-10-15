import json
import re
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "openai_results_100.jsonl"


def normalize(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())

def classify_suspicious(text):
    reasons = []

    if "\n" in text or "\r" in text:
        reasons.append("newline")

    if "____" in text:
        reasons.append("separator")

    suspicious_chars = {
        "<": "angle_bracket",
        ">": "angle_bracket",
        "[": "square_bracket",
        "]": "square_bracket",
        "@": "at_symbol",
        ";": "semicolon",
    }

    for char, reason in suspicious_chars.items():
        if char in text:
            reasons.append(reason)

    if "_" in text and "____" not in text:
        reasons.append("underscore")

    return reasons


def main():
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    all_fns = []

    for record in records:
        gold = {
            normalize(item["text"])
            for item in record["ground_truth"]
            if normalize(item["text"])
        }

        predicted = {
            normalize(item["original"])
            for item in record["entities_detected"]
            if normalize(item["original"])
        }

        fns = gold - predicted

        for fn in fns:
            original_text = next(
                item["text"]
                for item in record["ground_truth"]
                if normalize(item["text"]) == fn
            )

            reasons = classify_suspicious(original_text)

            all_fns.append({
                "document_id": record["document_id"],
                "text": original_text,
                "reasons": reasons,
            })

    print("=" * 100)
    print("SCREENING DEI FALSE NEGATIVE")
    print("=" * 100)

    print(f"\nTotale FN: {len(all_fns)}")

    suspicious = [
        item for item in all_fns
        if item["reasons"]
    ]

    clean = [
        item for item in all_fns
        if not item["reasons"]
    ]

    print(f"FN sospetti: {len(suspicious)}")
    print(f"FN senza indicatori evidenti: {len(clean)}")

    print("\n" + "-" * 100)
    print("MOTIVI DI SOSPETTO")
    print("-" * 100)

    reason_counter = Counter()

    for item in suspicious:
        for reason in item["reasons"]:
            reason_counter[reason] += 1

    for reason, count in reason_counter.most_common():
        print(f"{reason:20} {count}")

    print("\n" + "-" * 100)
    print("ELENCO FN SOSPETTI")
    print("-" * 100)

    for i, item in enumerate(suspicious, start=1):
        print(f"\n[{i}] {item['text']!r}")
        print(f"    Documento: {item['document_id']}")
        print(f"    Motivo: {', '.join(item['reasons'])}")

    print("\n" + "-" * 100)
    print("DOCUMENTI CON PIÙ FN")
    print("-" * 100)

    fn_by_doc = Counter(
        item["document_id"]
        for item in all_fns
    )

    for doc, count in fn_by_doc.most_common():
        print(f"{count:3}  {doc}")

    print("\n" + "=" * 100)
    print("FINE")
    print("=" * 100)


if __name__ == "__main__":
    main()