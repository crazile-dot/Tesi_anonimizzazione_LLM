
from pathlib import Path
import re
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "enronRandom-XML"
MANIFEST = PROJECT_DIR / "data" / "processed" / "email_manifest.csv"

TAG_PATTERN = re.compile(
    r"<true_name>(.*?)</true_name>",
    re.IGNORECASE | re.DOTALL
)

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    righe = list(csv.DictReader(f))

# Selezioniamo esempi di test con 1, 2 e almeno 5 annotazioni.
selezionati = []
categorie = [
    lambda n: n == 1,
    lambda n: n == 2,
    lambda n: n >= 5,
]

for criterio in categorie:
    for riga in righe:
        if riga["split"] != "test":
            continue

        try:
            conteggio = int(riga["annotation_count"])
        except (ValueError, KeyError):
            continue

        if criterio(conteggio):
            selezionati.append(riga)
            break

print("--- ESEMPI DI EMAIL ANNOTATE ---")

for riga in selezionati:
    percorso = DATA_DIR / Path(riga["relative_path"])
    testo = percorso.read_text(
        encoding="utf-8",
        errors="replace"
    )

    nomi = [
        " ".join(nome.split())
        for nome in TAG_PATTERN.findall(testo)
    ]

    print("\n" + "=" * 70)
    print(f"File: {riga['relative_path']}")
    print(f"Annotazioni: {len(nomi)}")
    print(f"Nomi annotati: {nomi}")

    # Mostriamo soltanto il contesto iniziale, senza stampare
    # l'intero messaggio né eventuali dati personali superflui.
    testo_senza_tag = TAG_PATTERN.sub(
        lambda m: " ".join(m.group(1).split()),
        testo
    )
    print("\nEstratto iniziale:")
    print(testo_senza_tag[:700])
