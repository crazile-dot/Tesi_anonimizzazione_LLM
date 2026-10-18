
import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

AUDIT = (
    BASE_DIR / "data" / "processed"
    / "audit_precision_recall_f1_680.csv"
)

CASES = [
    "dasovich-j_all_documents_18375",
    "davis-d_all_documents_440",
    "kean-s_all_documents_9447",
    "badeer-r_all_documents_128",
]


def parse_list(value):
    if not value:
        return []
    return json.loads(value)


with AUDIT.open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    rows = {
        row["document_id"]: row
        for row in csv.DictReader(f)
    }

for document_id in CASES:
    row = rows.get(document_id)

    print("\n" + "=" * 80)
    print(document_id)
    print("=" * 80)

    if row is None:
        print("Documento non trovato")
        continue

    print("\nANNOTAZIONI ORIGINALI:")
    print(parse_list(row["annotazioni_gold"]))

    for method in ("M4", "M5"):
        print(f"\n--- {method} ---")
        print(
            "TP:", row[f"{method}_TP"],
            "| FP:", row[f"{method}_FP"],
            "| FN:", row[f"{method}_FN"],
        )

        print("FP (entità aggiuntive):")
        print(parse_list(row[f"{method}_FP_names"]))

        print("FN (entità annotate non riconosciute):")
        print(parse_list(row[f"{method}_FN_names"]))
