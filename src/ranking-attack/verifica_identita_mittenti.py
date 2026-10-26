
from pathlib import Path
import csv
import re
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT = PROJECT_DIR / "data" / "processed" / "email_manifest.csv"
OUTPUT = PROJECT_DIR / "data" / "processed" / "verifica_identita_mittenti.csv"


def normalizza(testo):
    return re.sub(r"[^a-z0-9]", "", testo.casefold())


# Raccoglie le email associate a ciascun nome annotato.
associazioni = defaultdict(lambda: {
    "train_mittente": set(),
    "test_mittente": set(),
    "train_annotato": set(),
    "test_annotato": set(),
})

with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for riga in reader:
        split = riga["split"].strip().casefold()
        mittente = riga["from_email"].strip().casefold()
        nomi = riga["annotated_names"].split("|")

        local_part = (
            mittente.split("@")[0]
            if "@" in mittente
            else ""
        )

        for nome in nomi:
            nome = nome.strip()
            if not nome:
                continue

            dati = associazioni[nome.casefold()]

            if split == "train":
                dati["train_annotato"].add(riga["document_id"])
            elif split == "test":
                dati["test_annotato"].add(riga["document_id"])

            # Corrispondenza euristica: nome completo e mittente.
            if normalizza(nome) == normalizza(local_part):
                if split == "train":
                    dati["train_mittente"].add(riga["document_id"])
                elif split == "test":
                    dati["test_mittente"].add(riga["document_id"])


with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    campi = [
        "nome",
        "email_train",
        "email_test",
        "menzioni_train",
        "menzioni_test",
    ]
    writer = csv.DictWriter(f, fieldnames=campi)
    writer.writeheader()

    for nome, dati in sorted(associazioni.items()):
        writer.writerow({
            "nome": nome,
            "email_train": len(dati["train_mittente"]),
            "email_test": len(dati["test_mittente"]),
            "menzioni_train": len(dati["train_annotato"]),
            "menzioni_test": len(dati["test_annotato"]),
        })

print(f"Nomi analizzati: {len(associazioni)}")
print(f"File creato: {OUTPUT}")
