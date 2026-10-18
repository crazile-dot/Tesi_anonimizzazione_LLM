
import csv
import json
import re
import html
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST = BASE_DIR / "data" / "processed" / "email_manifest.csv"
RAW_DIR = BASE_DIR / "data" / "raw" / "enronRandom-XML"

# IMPORTANTE: usa gli stessi file utilizzati per calcolare
# le metriche aggregate già riportate nella tesi.
FILES = {
    "M4": BASE_DIR / "data" / "processed" / "openai_results_680_m4_v2.jsonl",
    "M5": BASE_DIR / "data" / "processed" / "openai_results_680_m5.jsonl",
}

OUTPUT = (
    BASE_DIR / "data" / "processed"
    / "audit_precision_recall_f1_680.csv"
)


def normalize(value):
    value = html.unescape(str(value or ""))
    return re.sub(r"\s+", " ", value).strip().casefold()


def read_gold_names(relative_path):
    path = RAW_DIR / relative_path.replace("\\", "/")

    if not path.exists():
        raise FileNotFoundError(
            f"Documento originale non trovato: {path}"
        )

    text = path.read_text(
        encoding="utf-8", errors="replace"
    )

    names = re.findall(
        r"<true_name>(.*?)</true_name>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return {
        normalize(name)
        for name in names
        if normalize(name)
    }


def read_results(path):
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


def get_predicted_names(record, method):
    if (
        record.get("status") != "success"
        or not isinstance(record.get("entities_detected"), list)
    ):
        return None

    predicted = set()

    for entity in record["entities_detected"]:
        if not isinstance(entity, dict):
            continue

        replacement = str(
            entity.get("replacement", "")
        ).strip()

        if method == "M4":
            if not re.fullmatch(r"PERSON_\d+", replacement):
                continue
        else:
            if str(entity.get("category", "")).strip().upper() != "PERSON":
                continue

        original = normalize(entity.get("original", ""))

        if original:
            predicted.add(original)

    return predicted


def join_names(names):
    return json.dumps(
        sorted(names),
        ensure_ascii=False
    )


# Carica le annotazioni originali.
documents = {}

with MANIFEST.open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    for row in csv.DictReader(f):
        documents[row["document_id"]] = {
            "relative_path": row["relative_path"],
            "gold": read_gold_names(row["relative_path"]),
        }

# Carica gli output dei due metodi.
results = {}

for method, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"File risultati {method} non trovato: {path}"
        )

    results[method] = read_results(path)

rows = []

for document_id, doc in documents.items():
    gold = doc["gold"]

    row = {
        "document_id": document_id,
        "relative_path": doc["relative_path"],
        "annotazioni_gold": join_names(gold),
        "numero_gold": len(gold),
    }

    for method in ("M4", "M5"):
        record = results[method].get(document_id, {})
        predicted = get_predicted_names(record, method)

        if predicted is None:
            row[f"{method}_status"] = "excluded"
            for field in (
                "predicted", "TP_names", "FP_names", "FN_names"
            ):
                row[f"{method}_{field}"] = ""
            for field in ("TP", "FP", "FN"):
                row[f"{method}_{field}"] = ""
            continue

        tp_names = gold & predicted
        fp_names = predicted - gold
        fn_names = gold - predicted

        row[f"{method}_status"] = "success"
        row[f"{method}_predicted"] = join_names(predicted)
        row[f"{method}_TP_names"] = join_names(tp_names)
        row[f"{method}_FP_names"] = join_names(fp_names)
        row[f"{method}_FN_names"] = join_names(fn_names)

        row[f"{method}_TP"] = len(tp_names)
        row[f"{method}_FP"] = len(fp_names)
        row[f"{method}_FN"] = len(fn_names)

    # Differenze M5 - M4: positive = più errori per M5.
    for metric in ("FP", "FN"):
        m4 = row.get(f"M4_{metric}")
        m5 = row.get(f"M5_{metric}")

        if isinstance(m4, int) and isinstance(m5, int):
            row[f"delta_{metric}_M5_vs_M4"] = m5 - m4
        else:
            row[f"delta_{metric}_M5_vs_M4"] = ""

    rows.append(row)

# Ordina i casi con più differenze complessive tra i metodi.
rows.sort(
    key=lambda r: (
        abs(r["delta_FP_M5_vs_M4"])
        + abs(r["delta_FN_M5_vs_M4"])
        if isinstance(r["delta_FP_M5_vs_M4"], int)
        and isinstance(r["delta_FN_M5_vs_M4"], int)
        else -1
    ),
    reverse=True,
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Documenti elaborati: {len(rows)}")
print(f"Audit salvato in: {OUTPUT}")
print("\nPrime 10 differenze più marcate:")

for row in rows[:10]:
    print(
        row["document_id"],
        "| M4 FP/FN:",
        row["M4_FP"], row["M4_FN"],
        "| M5 FP/FN:",
        row["M5_FP"], row["M5_FN"],
    )
