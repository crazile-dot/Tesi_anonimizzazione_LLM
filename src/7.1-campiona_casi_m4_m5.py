
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw" / "enronRandom-XML"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

MANIFEST_FILE = PROCESSED_DIR / "email_manifest.csv"
BASELINE_FILE = PROCESSED_DIR / "dataset_680_baseline.jsonl"
M4_FILE = PROCESSED_DIR / "openai_results_680_m4_v2.jsonl"
M5_FILE = PROCESSED_DIR / "openai_results_680_m5_v2.jsonl"

OUTPUT_CSV = PROCESSED_DIR / "audit_casi_m4_m5_v2.csv"
OUTPUT_JSONL = PROCESSED_DIR / "audit_casi_m4_m5_dettaglio_v2.jsonl"

SEED = 42
MAX_PER_CATEGORIA = 5


def leggi_jsonl(percorso):
    risultati = {}

    with open(percorso, "r", encoding="utf-8") as f:
        for riga in f:
            if not riga.strip():
                continue

            record = json.loads(riga)
            document_id = record.get("document_id")

            if document_id:
                risultati[document_id] = record

    return risultati


def leggi_menzioni(relative_path):
    percorso = RAW_DIR / relative_path.replace("\\", "/")
    testo = percorso.read_text(encoding="utf-8", errors="replace")

    return [
        re.sub(r"\s+", " ", nome).strip()
        for nome in re.findall(
            r"<true_name>(.*?)</true_name>",
            testo,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if nome.strip()
    ]


def conta_occorrenze(nome, testo):
    pattern = r"(?<!\w)" + re.escape(nome) + r"(?!\w)"
    return len(re.findall(pattern, testo, flags=re.IGNORECASE))


def testo_anonymizzato(record):
    return (
        record.get("text_anonymized")
        or record.get("testo_anonymizzato")
        or ""
    )


def risultato_valido(record):
    if not record:
        return False

    status = str(record.get("status", "")).lower()

    if status in {"error", "failed", "failure"}:
        return False

    return bool(testo_anonymizzato(record))


def conta_residui(menzioni, testo):
    frequenze = Counter(menzioni)
    residui = 0

    for nome, frequenza_originale in frequenze.items():
        trovate = conta_occorrenze(nome, testo)
        residui += min(frequenza_originale, trovate)

    return residui


def testo_originale_da_baseline(record):
    return (
        record.get("testo_originale")
        or record.get("M0_originale")
        or ""
    )


def main():
    random.seed(SEED)

    with open(MANIFEST_FILE, "r", encoding="utf-8-sig", newline="") as f:
        manifest = {
            r["document_id"]: r
            for r in csv.DictReader(f)
        }

    baseline = leggi_jsonl(BASELINE_FILE)
    m4 = leggi_jsonl(M4_FILE)
    m5 = leggi_jsonl(M5_FILE)

    casi = []

    for document_id, info in manifest.items():
        record_m4 = m4.get(document_id)
        record_m5 = m5.get(document_id)

        if not risultato_valido(record_m4) or not risultato_valido(record_m5):
            continue

        menzioni = leggi_menzioni(info["relative_path"])

        if not menzioni:
            continue

        testo_m4 = testo_anonymizzato(record_m4)
        testo_m5 = testo_anonymizzato(record_m5)

        residui_m4 = conta_residui(menzioni, testo_m4)
        residui_m5 = conta_residui(menzioni, testo_m5)

        if residui_m4 > 0 and residui_m5 == 0:
            categoria = "A_M4_residui_M5_zero"
        elif residui_m4 > residui_m5 > 0:
            categoria = "B_M5_meno_residui"
        elif residui_m5 > residui_m4:
            categoria = "C_M5_piu_residui"
        else:
            continue

        nomi_residui_m4 = [
            nome for nome in dict.fromkeys(menzioni)
            if conta_occorrenze(nome, testo_m4) > 0
        ]
        nomi_residui_m5 = [
            nome for nome in dict.fromkeys(menzioni)
            if conta_occorrenze(nome, testo_m5) > 0
        ]

        casi.append({
            "document_id": document_id,
            "categoria": categoria,
            "occorrenze_annotate": len(menzioni),
            "residui_M4": residui_m4,
            "residui_M5": residui_m5,
            "nomi_residui_M4": " | ".join(nomi_residui_m4),
            "nomi_residui_M5": " | ".join(nomi_residui_m5),
            "nomi_annotati": " | ".join(dict.fromkeys(menzioni)),
            "testo_originale": testo_originale_da_baseline(
                baseline.get(document_id, {})
            ),
            "testo_M4": testo_m4,
            "testo_M5": testo_m5,
        })

    categorie = [
        "A_M4_residui_M5_zero",
        "B_M5_meno_residui",
        "C_M5_piu_residui",
    ]

    selezionati = []

    for categoria in categorie:
        gruppo = [c for c in casi if c["categoria"] == categoria]
        gruppo.sort(key=lambda c: c["document_id"])
        random.shuffle(gruppo)
        selezionati.extend(gruppo[:MAX_PER_CATEGORIA])

        print(f"{categoria}: {len(gruppo)} casi disponibili")

    # CSV riepilogativo: più comodo da filtrare
    campi = [
        "document_id",
        "categoria",
        "occorrenze_annotate",
        "residui_M4",
        "residui_M5",
        "nomi_residui_M4",
        "nomi_residui_M5",
        "nomi_annotati",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campi)
        writer.writeheader()
        writer.writerows([
            {campo: caso[campo] for campo in campi}
            for caso in selezionati
        ])

    # JSONL con i testi completi per l'analisi manuale
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for caso in selezionati:
            f.write(json.dumps(caso, ensure_ascii=False) + "\n")

    print(f"\nCasi selezionati: {len(selezionati)}")
    print(f"Riepilogo: {OUTPUT_CSV}")
    print(f"Testi completi: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
