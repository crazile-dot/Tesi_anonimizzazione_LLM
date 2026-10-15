
from pathlib import Path
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent

CANDIDATI = PROJECT_DIR / "data" / "processed" / "candidati_attacco_filtrati.csv"
VERIFICA = PROJECT_DIR / "data" / "processed" / "verifica_identita_mittenti.csv"

with CANDIDATI.open("r", encoding="utf-8-sig", newline="") as f:
    candidati = list(csv.DictReader(f))

with VERIFICA.open("r", encoding="utf-8-sig", newline="") as f:
    verifiche = {
        r["nome"].strip().casefold(): r
        for r in csv.DictReader(f)
    }

print("CANDIDATO | TRAIN: email totali / corrispondenze mittente | TEST: email totali / corrispondenze mittente")

for c in candidati:
    nome = c["nome"].strip()
    v = verifiche.get(nome.casefold())

    if v:
        print(
            f"{nome} | "
            f"TRAIN: {c['email_train']} / {v['email_train']} | "
            f"TEST: {c['email_test']} / {v['email_test']}"
        )
