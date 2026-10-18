
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"

M4_FILE = DATA / "openai_results_680_m4_v2.jsonl"
M5_FILE = DATA / "openai_results_680_m5.jsonl"

# Casi scelti dall'audit: esempi di residui condivisi,
# residui solo M4 e residui solo M5.
CASES = [
    ("nemec-g_all_documents_2677", "Bracewell"),
    ("badeer-r_all_documents_39", "Vice Chairman"),
    ("saibi-e_all_documents_11", "Jean Ryall"),
    ("davis-d_all_documents_192", "Eve"),
    ("dasovich-j_all_documents_18375", "Bill"),
    ("kean-s_all_documents_9103", "Jimmy"),
    ("hodge-j_all_documents_228",
     "Janet Dietrich Ed McMichael Scott Neal Phillip Allen"),
]


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return {
            r["document_id"]: r
            for line in f
            if line.strip()
            for r in [json.loads(line)]
        }


def contexts(text, phrase, window=100):
    words = re.split(r"\s+", phrase.strip())
    if not phrase.strip():
        return [], 0

    pattern = r"(?<!\w)" + r"\s+".join(
        re.escape(word) for word in words
    ) + r"(?!\w)"

    matches = list(re.finditer(pattern, text))    
    result = []

    for match in matches[:5]:
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        result.append(
            text[start:match.start()]
            + "[["
            + text[match.start():match.end()]
            + "]]"
            + text[match.end():end]
        )

    return result, len(matches)


def main():
    m4 = load_jsonl(M4_FILE)
    m5 = load_jsonl(M5_FILE)

    for doc_id, phrase in CASES:
        print("\n" + "=" * 80)
        print(f"DOCUMENTO: {doc_id} | ANNOTAZIONE: {phrase!r}")

        for label, records in [("M4", m4), ("M5", m5)]:
            record = records.get(doc_id)

            if not record or record.get("status") != "success":
                print(f"\n{label}: risultato non disponibile")
                continue

            print(f"\n--- {label} ---")

            print("Entità rilevate corrispondenti:")
            found = [
                e for e in record.get("entities_detected", [])
                if phrase.casefold() in
                   e.get("original", "").casefold()
            ]

            for entity in found:
                print(" ", entity)

            text = record.get("text_anonymized") or ""
            result = contexts(text, phrase)

            excerpts, total = result

            print(f"Occorrenze nel testo anonimizzato: {total}")

            for excerpt in excerpts:
                print(" ", repr(excerpt))

            if total == 0:
                print("Nessuna occorrenza letterale nel testo anonimizzato.")


if __name__ == "__main__":
    main()
