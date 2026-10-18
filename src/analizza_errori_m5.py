
import csv
import json
import html
import re
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST = BASE_DIR / "data" / "processed" / "email_manifest.csv"
RAW_DIR = BASE_DIR / "data" / "raw" / "enronRandom-XML"

M4_FILE = BASE_DIR / "data" / "processed" / "openai_results_680_m4_v2.jsonl"
M5_FILE = BASE_DIR / "data" / "processed" / "openai_results_680_m5.jsonl"
OUTPUT = BASE_DIR / "data" / "processed" / "campione_errori_m5.csv"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def norm(value):
    value = html.unescape(str(value or ""))
    return re.sub(r"\s+", " ", value).strip().casefold()


def read_jsonl(path):
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("document_id"):
                    records[r["document_id"]] = r
    return records


def get_gold(relative_path):
    path = RAW_DIR / relative_path.replace("\\", "/")
    text = path.read_text(encoding="utf-8", errors="replace")
    names = re.findall(
        r"<true_name>(.*?)</true_name>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return {norm(n) for n in names if norm(n)}


def get_predictions(record):
    entities = record.get("entities_detected", [])
    return {
        norm(e.get("original", ""))
        for e in entities
        if isinstance(e, dict)
        and norm(e.get("original", ""))
        and not EMAIL_RE.fullmatch(norm(e.get("original", "")))
    }


m4 = read_jsonl(M4_FILE)
m5 = read_jsonl(M5_FILE)

rows = []
rng = random.Random(42)

with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
    manifest = list(csv.DictReader(f))

for row in manifest:
    doc_id = row["document_id"]
    r4, r5 = m4.get(doc_id, {}), m5.get(doc_id, {})

    if r4.get("status") != "success" or r5.get("status") != "success":
        continue

    gold = get_gold(row["relative_path"])
    p4 = get_predictions(r4)
    p5 = get_predictions(r5)

    original_text = html.unescape(r5.get("text_original", ""))
    anonymized_text = html.unescape(r5.get("text_anonymized", ""))

    # Falsi positivi M5: predizioni non presenti nel riferimento XML.
    for entity in p5 - gold:
        rows.append({
            "tipo_errore": "FP_M5",
            "document_id": doc_id,
            "entita": entity,
            "presente_in_gold": False,
            "rilevata_da_M4": entity in p4,
            "testo_originale": original_text[:1200],
            "testo_anonymizzato_M5": anonymized_text[:1200],
        })

    # Falsi negativi M5: entità gold non rilevate da M5.
    for entity in gold - p5:
        rows.append({
            "tipo_errore": "FN_M5",
            "document_id": doc_id,
            "entita": entity,
            "presente_in_gold": True,
            "rilevata_da_M4": entity in p4,
            "testo_originale": original_text[:1200],
            "testo_anonymizzato_M5": anonymized_text[:1200],
        })

selected = []
for error_type in ("FP_M5", "FN_M5"):
    candidates = [r for r in rows if r["tipo_errore"] == error_type]
    selected.extend(rng.sample(candidates, min(30, len(candidates))))

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "tipo_errore", "document_id", "entita",
        "presente_in_gold", "rilevata_da_M4",
        "testo_originale", "testo_anonymizzato_M5",
    ])
    writer.writeheader()
    writer.writerows(selected)

print(f"Campione salvato: {OUTPUT}")
print(f"Falsi positivi M5 disponibili: {sum(r['tipo_errore'] == 'FP_M5' for r in rows)}")
print(f"Falsi negativi M5 disponibili: {sum(r['tipo_errore'] == 'FN_M5' for r in rows)}")
print(f"Righe estratte: {len(selected)}")
print("Apri il CSV e confronta entità, contesto originale e rilevamento M4.")
