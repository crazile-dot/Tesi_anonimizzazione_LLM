from pathlib import Path
import csv
import re
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

MANIFEST = DATA_DIR / "email_manifest.csv"
RIFERIMENTI = DATA_DIR / "profili_riferimento_ampliato.csv"
OUTPUT = DATA_DIR / "candidati_attacco.csv"


def normalizza(testo):
    return re.sub(r"[^a-z0-9]", "", testo.casefold())


# 1. Recupera l'insieme dei candidati dai riferimenti TRAIN.
# Questa è la stessa selezione usata per costruire i profili.
candidati = {}

with RIFERIMENTI.open("r", encoding="utf-8-sig", newline="") as f:
    for riga in csv.DictReader(f):
        nome = riga["candidato"].strip()
        chiave = normalizza(nome)

        if not chiave:
            continue

        candidati[chiave] = nome


# 2. Conta i documenti TRAIN e TEST inviati da ciascun candidato.
conteggi = defaultdict(lambda: {"train": 0, "test": 0})

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    for riga in csv.DictReader(f):
        split = riga["split"].strip().casefold()

        if split not in {"train", "test"}:
            continue

        mittente = riga["from_email"].strip().casefold()

        if "@" not in mittente:
            continue

        local_part = mittente.split("@", 1)[0]
        chiave = normalizza(local_part)

        if chiave not in candidati:
            continue

        conteggi[chiave][split] += 1


# 3. Produce l'elenco dei candidati e mantiene anche quelli
# che non hanno email TEST: la loro copertura TEST sarà zero.
righe = []

for chiave, nome in candidati.items():
    righe.append({
        "nome": nome,
        "email_train_riferimento": sum(
            1
            for _ in []
        ),
        "email_train_mittente": conteggi[chiave]["train"],
        "email_test_mittente": conteggi[chiave]["test"],
    })


# 4. Conta le email di riferimento effettivamente usate
# per costruire ogni profilo, senza confonderle con tutte
# le email inviate dal mittente nel TRAIN.
riferimenti_per_candidato = defaultdict(int)

with RIFERIMENTI.open("r", encoding="utf-8-sig", newline="") as f:
    for riga in csv.DictReader(f):
        riferimenti_per_candidato[riga["candidato"]] += 1

for riga in righe:
    riga["email_train_riferimento"] = riferimenti_per_candidato[
        riga["nome"]
    ]


righe.sort(
    key=lambda x: (
        x["email_train_riferimento"],
        x["email_test_mittente"]
    ),
    reverse=True
)

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    campi = [
        "nome",
        "email_train_riferimento",
        "email_train_mittente",
        "email_test_mittente",
    ]

    writer = csv.DictWriter(f, fieldnames=campi)
    writer.writeheader()
    writer.writerows(righe)


print(f"Candidati con profilo TRAIN: {len(righe)}")
print(
    "Candidati con almeno un mittente TEST corrispondente: "
    f"{sum(1 for r in righe if r['email_test_mittente'] > 0)}"
)
print(f"File creato: {OUTPUT}")

print("\nPrime 20 entità candidate:")

for riga in righe[:20]:
    print(
        f"{riga['nome']} | "
        f"riferimenti TRAIN: {riga['email_train_riferimento']} | "
        f"email TRAIN mittente: {riga['email_train_mittente']} | "
        f"email TEST mittente: {riga['email_test_mittente']}"
    )
