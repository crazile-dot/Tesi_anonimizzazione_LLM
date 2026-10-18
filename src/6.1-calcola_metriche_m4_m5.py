
import csv
import json
import re
import html
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST = BASE_DIR / "data" / "processed" / "email_manifest.csv"
RAW_DIR = BASE_DIR / "data" / "raw" / "enronRandom-XML"

FILES = {
    "M4": BASE_DIR / "data" / "processed" / "openai_results_680_m4_v2.jsonl",
    "M5": BASE_DIR / "data" / "processed" / "openai_results_680_m5_v2.jsonl",
}

OUTPUT = BASE_DIR / "data" / "processed" / "metriche_precision_recall_f1_m4_m5_v2.csv"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize(value):
    """Normalizza maiuscole e spazi per confrontare le entità."""
    value = html.unescape(str(value or ""))
    return re.sub(r"\s+", " ", value).strip().casefold()


def read_gold_names(relative_path):
    """Estrae le annotazioni true_name dal documento XML originale."""
    path = RAW_DIR / relative_path.replace("\\", "/")

    if not path.exists():
        raise FileNotFoundError(f"Documento originale non trovato: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")

    names = re.findall(
        r"<true_name>(.*?)</true_name>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return {normalize(name) for name in names if normalize(name)}


def read_results(path):
    """Legge i risultati JSONL indicizzati per document_id."""
    results = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            document_id = record.get("document_id")

            if document_id:
                results[document_id] = record

    return results


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


# Legge il manifest e prepara il riferimento annotato.
gold_by_id = {}

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        document_id = row["document_id"]
        relative_path = row["relative_path"]
        gold_by_id[document_id] = read_gold_names(relative_path)

print(f"Documenti nel manifest: {len(gold_by_id)}")

all_results = {}
for method, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Risultati {method} non trovati: {path}")
    all_results[method] = read_results(path)

rows = []

for method, results in all_results.items():
    tp = fp = fn = 0
    evaluated = 0
    skipped = 0

    for document_id, gold in gold_by_id.items():
        record = results.get(document_id, {})

        if (
            record.get("status") != "success"
            or not isinstance(record.get("entities_detected"), list)
        ):
            skipped += 1
            continue

        
        
        predicted = set()

        for entity in record["entities_detected"]:
            if not isinstance(entity, dict):
                continue

            replacement = str(entity.get("replacement", "")).strip()

            if not re.fullmatch(r"PERSON_\d+", replacement):
                continue

            original = normalize(entity.get("original", ""))

            if original:
                predicted.add(original)

        doc_tp = len(gold & predicted)
        doc_fp = len(predicted - gold)
        doc_fn = len(gold - predicted)

        tp += doc_tp
        fp += doc_fp
        fn += doc_fn
        evaluated += 1

    precision, recall, f1 = calculate_metrics(tp, fp, fn)

    row = {
        "metodo": method,
        "documenti_valutati": evaluated,
        "documenti_esclusi": skipped,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    rows.append(row)

    print(f"\n{'=' * 45}")
    print(f"METODO {method}")
    print(f"{'=' * 45}")
    print(f"Documenti valutati: {evaluated}")
    print(f"Documenti esclusi:  {skipped}")
    print(f"TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"Precision: {precision:.4f} ({precision:.2%})")
    print(f"Recall:    {recall:.4f} ({recall:.2%})")
    print(f"F1-score:  {f1:.4f} ({f1:.2%})")

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nRisultati salvati in:\n{OUTPUT}")
