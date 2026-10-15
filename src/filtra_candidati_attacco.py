
from pathlib import Path
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT = PROJECT_DIR / "data" / "processed" / "candidati_attacco.csv"
OUTPUT = PROJECT_DIR / "data" / "processed" / "candidati_attacco_filtrati.csv"

MIN_TRAIN = 3
MIN_TEST = 1

candidati_filtrati = []

with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for riga in reader:
        nome = riga["nome"].strip()
        train = int(riga["email_train"])
        test = int(riga["email_test"])

        if (
            len(nome.split()) >= 2
            and train >= MIN_TRAIN
            and test >= MIN_TEST
        ):
            candidati_filtrati.append({
                "nome": nome,
                "email_train": train,
                "email_test": test
            })

candidati_filtrati.sort(
    key=lambda x: (x["email_train"], x["email_test"]),
    reverse=True
)

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    campi = ["nome", "email_train", "email_test"]
    writer = csv.DictWriter(f, fieldnames=campi)
    writer.writeheader()
    writer.writerows(candidati_filtrati)

print(f"Candidati iniziali selezionati: {len(candidati_filtrati)}")
print(f"File creato: {OUTPUT}")
print("\nPrime 20 entità:")

for c in candidati_filtrati[:20]:
    print(
        f"{c['nome']} | train: {c['email_train']} | "
        f"test: {c['email_test']}"
    )
