
import csv
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"

INPUT_FILE = DATA / "campione_revisione_residui_m4_m5.csv"
M4_FILE = DATA / "openai_results_680_m4_v2.jsonl"
M5_FILE = DATA / "openai_results_680_m5.jsonl"
OUTPUT_FILE = DATA / "campione_revisione_con_contesti.csv"

WINDOW = 180
MAX_CONTEXTS = 3


def load_jsonl(path):
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records[record["document_id"]] = record
    return records


def find_contexts(text, phrase):
    phrase = " ".join(phrase.split())
    if not phrase:
        return []

    words = re.split(r"\s+", phrase)
    pattern = (
        r"(?<!\w)"
        + r"\s+".join(re.escape(word) for word in words)
        + r"(?!\w)"
    )

    contexts = []
    for match in list(re.finditer(pattern, text))[:MAX_CONTEXTS]:
        start = max(0, match.start() - WINDOW)
        end = min(len(text), match.end() + WINDOW)
        excerpt = text[start:end]
        excerpt = excerpt.replace("\r", " ").replace("\n", " ")
        contexts.append(excerpt)

    return contexts


def entity_summary(record, phrase):
    entities = record.get("entities_detected", [])
    matches = [
        e for e in entities
        if phrase.casefold() in e.get("original", "").casefold()
        or e.get("original", "").casefold() in phrase.casefold()
    ]

    if not matches:
        return "Nessuna entità corrispondente rilevata"

    return " | ".join(
        f"{e.get('original', '')} -> {e.get('replacement', '')} "
        f"[{e.get('category', '')}]"
        for e in matches
    )


def main():
    m4 = load_jsonl(M4_FILE)
    m5 = load_jsonl(M5_FILE)

    with INPUT_FILE.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    extra_fields = [
        "entita_rilevate_m4",
        "contesti_residui_m4",
        "entita_rilevate_m5",
        "contesti_residui_m5",
        "classificazione_finale",
        "note_revisione",
    ]

    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    for row in rows:
        doc_id = row["document_id"]
        phrase = row["annotazione_xml"]

        r4 = m4.get(doc_id, {})
        r5 = m5.get(doc_id, {})

        text4 = r4.get("text_anonymized") or ""
        text5 = r5.get("text_anonymized") or ""

        row["entita_rilevate_m4"] = entity_summary(r4, phrase)
        row["entita_rilevate_m5"] = entity_summary(r5, phrase)

        row["contesti_residui_m4"] = " || ".join(
            find_contexts(text4, phrase)
        )
        row["contesti_residui_m5"] = " || ".join(
            find_contexts(text5, phrase)
        )

        if not row.get("classificazione_finale"):
            row["classificazione_finale"] = ""
        if not row.get("note_revisione"):
            row["note_revisione"] = ""

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Casi arricchiti: {len(rows)}")
    print(f"File creato: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
