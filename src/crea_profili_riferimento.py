
from pathlib import Path
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT = PROJECT_DIR / "data" / "processed" / "email_manifest.csv"
OUTPUT = PROJECT_DIR / "data" / "processed" / "profili_riferimento.csv"

CANDIDATI = {
    "scott neal": "scott.neal@enron.com",
    "errol mclaughlin": "errol.mclaughlin@enron.com",
    "drew fossum": "drew.fossum@enron.com",
    "mike mcconnell": "mike.mcconnell@enron.com",
}

righe = []

with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
    for doc in csv.DictReader(f):
        if doc["split"].strip().casefold() != "train":
            continue

        mittente = doc["from_email"].strip().casefold()

        for candidato, email in CANDIDATI.items():
            if mittente == email:
                righe.append({
                    "candidato": candidato,
                    "document_id": doc["document_id"],
                    "relative_path": doc["relative_path"],
                    "email_mittente": mittente,
                })

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    campi = [
        "candidato",
        "document_id",
        "relative_path",
        "email_mittente",
    ]
    writer = csv.DictWriter(f, fieldnames=campi)
    writer.writeheader()
    writer.writerows(righe)

print(f"Documenti di riferimento selezionati: {len(righe)}")
print(f"File creato: {OUTPUT}")

for candidato in CANDIDATI:
    documenti = [r for r in righe if r["candidato"] == candidato]
    print(f"{candidato}: {len(documenti)} email train")
