
from pathlib import Path
import csv
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "enronRandom-XML"
RIFERIMENTI = PROJECT_DIR / "data" / "processed" / "profili_riferimento.csv"
OUTPUT = PROJECT_DIR / "data" / "processed" / "profili_testuali.csv"


def leggi_testo(percorso):
    testo = percorso.read_text(encoding="utf-8", errors="replace")
    testo = re.sub(r"<true_name>(.*?)</true_name>", r"\1", testo, flags=re.I | re.S)
    return testo


profili = {}

with RIFERIMENTI.open("r", encoding="utf-8-sig", newline="") as f:
    riferimenti = list(csv.DictReader(f))

for riga in riferimenti:
    candidato = riga["candidato"]
    percorso = RAW_DIR / Path(riga["relative_path"])

    if not percorso.exists():
        print(f"ATTENZIONE: file non trovato: {percorso}")
        continue

    testo = leggi_testo(percorso)
    profili.setdefault(candidato, []).append(testo)

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    campi = ["candidato", "numero_email", "profilo_testuale"]
    writer = csv.DictWriter(f, fieldnames=campi)
    writer.writeheader()

    for candidato, testi in profili.items():
        writer.writerow({
            "candidato": candidato,
            "numero_email": len(testi),
            "profilo_testuale": "\n\n".join(testi),
        })

print(f"Profili creati: {len(profili)}")
print(f"File creato: {OUTPUT}")

for candidato, testi in profili.items():
    print(f"{candidato}: {len(testi)} email, {sum(len(t) for t in testi)} caratteri")
