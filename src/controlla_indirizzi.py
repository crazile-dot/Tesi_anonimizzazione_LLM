
from pathlib import Path
from collections import Counter
import csv
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "enronRandom-XML"
MANIFEST = PROJECT_DIR / "data" / "processed" / "email_manifest.csv"

TAG_PATTERN = re.compile(
    r"<true_name>(.*?)</true_name>",
    re.IGNORECASE | re.DOTALL
)
EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE
)

totali = Counter()
esempi = []

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row["split"] != "test":
            continue

        percorso = DATA_DIR / Path(row["relative_path"])
        testo = percorso.read_text(
            encoding="utf-8",
            errors="replace"
        )

        nomi = [
            " ".join(n.split()).casefold()
            for n in TAG_PATTERN.findall(testo)
        ]
        indirizzi = {
            e.casefold() for e in EMAIL_PATTERN.findall(testo)
        }

        # Cerchiamo una corrispondenza semplice tra nome annotato
        # e parte locale dell'indirizzo, senza considerarla una prova
        # certa dell'identità.
        corrispondenze = []
        for nome in nomi:
            parole = re.findall(r"[a-z0-9]+", nome)
            if len(parole) < 2:
                continue

            for indirizzo in indirizzi:
                parte_locale = indirizzo.split("@")[0]
                if all(parola in parte_locale for parola in parole):
                    corrispondenze.append((nome, indirizzo))

        totali["email_test"] += 1
        if nomi:
            totali["email_con_annotazioni"] += 1
        if indirizzi:
            totali["email_con_indirizzi"] += 1
        if corrispondenze:
            totali["email_con_corrispondenza_semplice"] += 1
            if len(esempi) < 10:
                esempi.append((row["relative_path"], corrispondenze))

print("--- RIEPILOGO TEST ---")
for chiave, valore in totali.items():
    print(f"{chiave}: {valore}")

print("\n--- ESEMPI DI CORRISPONDENZA ---")
for percorso, corrispondenze in esempi:
    print(f"\n{percorso}")
    for nome, indirizzo in corrispondenze[:5]:
        print(f"  {nome!r} <-> {indirizzo}")
