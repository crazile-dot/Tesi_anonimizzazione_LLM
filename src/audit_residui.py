
import csv
import json
import re
from pathlib import Path
from collections import Counter

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"
RAW = PROJECT / "data" / "raw" / "enronRandom-XML"

MANIFEST = DATA / "email_manifest.csv"
M4_FILE = DATA / "openai_results_680_m4_v2.jsonl"
M5_FILE = DATA / "openai_results_680_m5.jsonl"

OUTPUT_TXT = DATA / "audit_residui_m4_m5_680.txt"
OUTPUT_CSV = DATA / "audit_residui_m4_m5_680.csv"

TAG_PATTERN = re.compile(
    r"<true_name>(.*?)</true_name>",
    re.IGNORECASE | re.DOTALL,
)


def load_jsonl(path):
    records = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records[record["document_id"]] = record

    return records


def count_occurrences(text, phrase):
    """Conta le occorrenze letterali, case-sensitive,
    rispettando i confini delle parole e gli spazi tra parole.
    """
    phrase = " ".join(phrase.split())

    if not phrase:
        return 0

    words = re.split(r"\s+", phrase)

    pattern = (
        r"(?<!\w)"
        + r"\s+".join(re.escape(word) for word in words)
        + r"(?!\w)"
    )

    return len(re.findall(pattern, text))


def normalize_annotation(text):
    """Normalizza gli spazi senza ignorare maiuscole/minuscole."""
    return " ".join(text.split())


def main():
    with MANIFEST.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        manifest = list(csv.DictReader(f))

    m4 = load_jsonl(M4_FILE)
    m5 = load_jsonl(M5_FILE)

    # Chiave: documento + testo esatto dell'annotazione normalizzata.
    # Manteniamo separate annotazioni diverse, anche se simili.
    cases = {}

    docs_evaluated = 0
    docs_skipped = 0
    annotation_count = 0

    for row in manifest:
        doc_id = row["document_id"]
        r4 = m4.get(doc_id, {})
        r5 = m5.get(doc_id, {})

        if (
            r4.get("status") != "success"
            or r5.get("status") != "success"
        ):
            docs_skipped += 1
            continue

        raw_path = RAW / Path(row["relative_path"])

        if not raw_path.exists():
            print(f"ATTENZIONE: file XML non trovato: {raw_path}")
            docs_skipped += 1
            continue

        raw_text = raw_path.read_text(
            encoding="latin-1",
            errors="replace",
        )

        annotations = [
            normalize_annotation(name)
            for name in TAG_PATTERN.findall(raw_text)
            if name.strip()
        ]

        text4 = r4.get("text_anonymized") or ""
        text5 = r5.get("text_anonymized") or ""

        docs_evaluated += 1

        for annotation in annotations:
            annotation_count += 1

            residual_m4 = count_occurrences(text4, annotation)
            residual_m5 = count_occurrences(text5, annotation)

            key = (doc_id, annotation)

            # Se la stessa annotazione compare più volte nello stesso
            # documento, accumuliamo le occorrenze senza duplicare il caso.
            if key not in cases:
                cases[key] = {
                    "document_id": doc_id,
                    "annotazione_xml": annotation,
                    "occorrenze_residue_m4": 0,
                    "occorrenze_residue_m5": 0,
                    "numero_annotazioni_xml": 0,
                }

            cases[key]["occorrenze_residue_m4"] += residual_m4
            cases[key]["occorrenze_residue_m5"] += residual_m5
            cases[key]["numero_annotazioni_xml"] += 1

    # Classificazione per coppia documento-annotazione.
    category_counts = Counter()
    detailed_cases = []

    for item in cases.values():
        m4_count = item["occorrenze_residue_m4"]
        m5_count = item["occorrenze_residue_m5"]

        if m4_count > 0 and m5_count > 0:
            category = "residuo_entrambi"
        elif m4_count > 0:
            category = "solo_M4"
        elif m5_count > 0:
            category = "solo_M5"
        else:
            category = "nessun_residuo"

        item["categoria"] = category
        detailed_cases.append(item)
        category_counts[category] += 1

    # Ordine deterministico per facilitare controlli e confronti.
    detailed_cases.sort(
        key=lambda item: (
            item["document_id"],
            item["annotazione_xml"],
        )
    )

    fields = [
        "document_id",
        "annotazione_xml",
        "numero_annotazioni_xml",
        "occorrenze_residue_m4",
        "occorrenze_residue_m5",
        "categoria",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for item in detailed_cases:
            writer.writerow({
                field: item[field]
                for field in fields
            })

    # Statistiche complessive sulle annotazioni XML.
    # Per ogni annotazione ripetuta, valutiamo la relativa occorrenza
    # nel documento, come fa verifica.py.
    total_annotations = 0
    protected_m4 = 0
    protected_m5 = 0
    residual_annotations_m4 = 0
    residual_annotations_m5 = 0
    residual_occurrences_m4 = 0
    residual_occurrences_m5 = 0

    for row in manifest:
        doc_id = row["document_id"]
        r4 = m4.get(doc_id, {})
        r5 = m5.get(doc_id, {})

        if (
            r4.get("status") != "success"
            or r5.get("status") != "success"
        ):
            continue

        raw_path = RAW / Path(row["relative_path"])

        if not raw_path.exists():
            continue

        raw_text = raw_path.read_text(
            encoding="latin-1",
            errors="replace",
        )

        annotations = [
            normalize_annotation(name)
            for name in TAG_PATTERN.findall(raw_text)
            if name.strip()
        ]

        text4 = r4.get("text_anonymized") or ""
        text5 = r5.get("text_anonymized") or ""

        for annotation in annotations:
            total_annotations += 1

            count4 = count_occurrences(text4, annotation)
            count5 = count_occurrences(text5, annotation)

            residual_occurrences_m4 += count4
            residual_occurrences_m5 += count5

            if count4 == 0:
                protected_m4 += 1
            else:
                residual_annotations_m4 += 1

            if count5 == 0:
                protected_m5 += 1
            else:
                residual_annotations_m5 += 1

    recall_m4 = (
        protected_m4 / total_annotations
        if total_annotations else 0
    )
    recall_m5 = (
        protected_m5 / total_annotations
        if total_annotations else 0
    )

    lines = [
        "AUDIT RESIDUI M4 vs M5",
        "Criterio: case-sensitive, parole complete, spazi normalizzati.",
        "Il residuo letterale non equivale automaticamente a rischio "
        "di reidentificazione.",
        "",
        f"Documenti valutati: {docs_evaluated}",
        f"Documenti saltati: {docs_skipped}",
        f"Annotazioni XML valutate: {total_annotations}",
        "",
        "--- METRICHE PER ANNOTAZIONE XML ---",
        f"M4 - senza residui: {protected_m4}",
        f"M4 - con residui: {residual_annotations_m4}",
        f"M4 - occorrenze residue: {residual_occurrences_m4}",
        f"M4 - recall di protezione: {recall_m4:.4%}",
        "",
        f"M5 - senza residui: {protected_m5}",
        f"M5 - con residui: {residual_annotations_m5}",
        f"M5 - occorrenze residue: {residual_occurrences_m5}",
        f"M5 - recall di protezione: {recall_m5:.4%}",
        "",
        "--- CONFRONTO PER CASO DOCUMENTO-ANNOTAZIONE ---",
        f"Casi distinti: {len(detailed_cases)}",
        f"Residuo in entrambi: {category_counts['residuo_entrambi']}",
        f"Residuo solo M4: {category_counts['solo_M4']}",
        f"Residuo solo M5: {category_counts['solo_M5']}",
        f"Nessun residuo: {category_counts['nessun_residuo']}",
        "",
        f"CSV dettagliato: {OUTPUT_CSV}",
    ]

    report = "\n".join(lines)
    print(report)

    OUTPUT_TXT.write_text(report, encoding="utf-8")
    print(f"\nReport salvato in: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
