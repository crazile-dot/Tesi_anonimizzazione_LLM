
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

TAG_PATTERN = re.compile(
    r"<true_name>(.*?)</true_name>",
    re.IGNORECASE | re.DOTALL,
)


def normalize(text):
    return " ".join(text.split()).casefold()


def load_results(path):
    results = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                results[record["document_id"]] = record
    return results



def count_occurrences(text, phrase):
    """Conta le occorrenze come parole o sequenze di parole complete."""
    phrase = " ".join(phrase.split())

    if not phrase:
        return 0

    parole = re.split(r"\s+", phrase)
    pattern = r"(?<!\w)" + r"\s+".join(
        re.escape(parola) for parola in parole
    ) + r"(?!\w)"

    return len(re.findall(pattern, text))

def main():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))

    m4 = load_results(M4_FILE)
    m5 = load_results(M5_FILE)

    stats = {
        "M4": Counter(),
        "M5": Counter(),
    }
    examples = {
        "M4": [],
        "M5": [],
    }

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
        raw_text = raw_path.read_text(
            encoding="latin-1",
            errors="replace",
        )

        annotations = [
            " ".join(name.split())
            for name in TAG_PATTERN.findall(raw_text)
        ]

        for method, record in [("M4", r4), ("M5", r5)]:
            anonymized = record.get("text_anonymized") or ""

            for name in annotations:
                if not name.strip():
                    continue

                stats[method]["annotazioni_xml"] += 1
                residuals = count_occurrences(anonymized, name)

                if residuals == 0:
                    stats[method]["annotazioni_senza_residui"] += 1
                else:
                    stats[method]["annotazioni_con_residui"] += 1
                    stats[method]["occorrenze_residue"] += residuals

                    if len(examples[method]) < 15:
                        examples[method].append({
                            "document_id": doc_id,
                            "annotazione": name,
                            "occorrenze_residue": residuals,
                        })

    output_lines = []

    for method in ("M4", "M5"):
        s = stats[method]
        total = s["annotazioni_xml"]
        protected = s["annotazioni_senza_residui"]
        residual = s["annotazioni_con_residui"]
        recall = protected / total if total else 0

        output_lines.extend([
            f"\n=== {method} ===",
            f"Annotazioni XML valutate: {total}",
            f"Annotazioni senza residui letterali: {protected}",
            f"Annotazioni con almeno un residuo: {residual}",
            f"Occorrenze residue complessive: {s['occorrenze_residue']}",
            f"Recall di protezione (per annotazione): {recall:.4%}",
            "\nEsempi di residui:",
        ])

        for item in examples[method]:
            output_lines.append(
                f"- {item['document_id']}: "
                f"{item['annotazione']!r} "
                f"({item['occorrenze_residue']} occorrenze)"
            )

    report = "\n".join(output_lines)
    print(report)

    output_file = DATA / "residui_annotazioni_m4_m5_680.txt"
    output_file.write_text(report, encoding="utf-8")
    print(f"\nReport salvato in: {output_file}")


if __name__ == "__main__":
    main()




