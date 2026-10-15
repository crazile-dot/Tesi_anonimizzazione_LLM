
from pathlib import Path
import csv
import json
import re

from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "enronRandom-XML"
MANIFEST = PROJECT_DIR / "data" / "processed" / "email_manifest.csv"
OUTPUT = PROJECT_DIR / "data" / "processed" / "corpus_test.jsonl"

TAG_PATTERN = re.compile(
    r"<true_name>(.*?)</true_name>",
    re.IGNORECASE | re.DOTALL
)

HEADER_BODY_SEPARATOR = re.compile(r"\r?\n\r?\n")

documenti = []

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row["split"] != "test":
            continue

        percorso = DATA_DIR / Path(row["relative_path"])

        testo = percorso.read_text(
            encoding="utf-8",
            errors="replace"
        )

        separatore = HEADER_BODY_SEPARATOR.search(testo)

        if separatore:
            header = testo[:separatore.start()]
            corpo = testo[separatore.end():]
        else:
            header = testo
            corpo = ""

        nomi_originali = [
            " ".join(nome.split())
            for nome in TAG_PATTERN.findall(testo)
        ]

        # Manteniamo il testo originale e le annotazioni
        # in un archivio separato, destinato alla valutazione.
        documento = {
            "document_id": row["relative_path"],
            "header_originale": header,
            "corpo_originale": corpo,
            "testo_originale": testo,
            "nomi_annotati": nomi_originali,
            "numero_annotazioni": len(nomi_originali),
            "numero_nomi_distinti": len(
                {nome.casefold() for nome in nomi_originali}
            ),
        }

        documenti.append(documento)

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("Documenti nel manifest:", len(rows))
print("Distribuzione degli split:")

for split, count in sorted(
    Counter(row["split"] for row in rows).items()
):
    print(f"  {split}: {count}")

print(
    "Documenti selezionati dallo script:",
    sum(row["split"] == "test" for row in rows)
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8", newline="\n") as f:
    for documento in documenti:
        f.write(
            json.dumps(documento, ensure_ascii=False) + "\n"
        )

print(f"Email elaborate: {len(documenti)}")
print(
    "Email con annotazioni: "
    f"{sum(bool(d['nomi_annotati']) for d in documenti)}"
)
print(
    "Email senza separatore header/corpo: "
    f"{sum(not d['corpo_originale'] for d in documenti)}"
)
print(f"Archivio creato: {OUTPUT}")
