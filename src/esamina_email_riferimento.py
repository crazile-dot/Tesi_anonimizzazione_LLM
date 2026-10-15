
from pathlib import Path
import csv
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT = PROJECT_DIR / "data" / "processed" / "email_manifest.csv"

CANDIDATI = [
    "scott neal",
    "errol mclaughlin",
    "drew fossum",
    "tana jones",
    "mike mcconnell",
]


def normalizza(testo):
    return re.sub(r"[^a-z0-9]", "", testo.casefold())


with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
    documenti = list(csv.DictReader(f))

for candidato in CANDIDATI:
    print(f"\n{'=' * 60}")
    print(f"CANDIDATO: {candidato}")

    nome_normalizzato = normalizza(candidato)

    for split in ("train", "test"):
        print(f"\n  {split.upper()}:")

        trovati = 0

        for doc in documenti:
            if doc["split"].strip().casefold() != split:
                continue

            mittente = doc["from_email"].strip()
            local_part = (
                mittente.split("@")[0]
                if "@" in mittente
                else ""
            )

            if normalizza(local_part) == nome_normalizzato:
                print(
                    f"    ID: {doc['document_id']} | "
                    f"Mittente: {mittente} | "
                    f"Percorso: {doc['relative_path']}"
                )
                trovati += 1

        if trovati == 0:
            print("    Nessuna corrispondenza esatta col mittente.")
