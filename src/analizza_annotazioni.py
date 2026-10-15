
from pathlib import Path
from collections import Counter
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "raw" / "enronRandom-XML"

# Escludiamo archivi ZIP e altri file non testuali del dataset.
files = [
    p for p in DATASET_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() != ".zip"
]

conteggi_annotazioni = Counter()
frequenze_nomi = Counter()
esempi = {}

for path in files:
    try:
        testo = path.read_text(
            encoding="utf-8",
            errors="replace"
        )
    except OSError:
        continue

    nomi = [
        nome.strip()
        for nome in re.findall(
            r"<true_name>(.*?)</true_name>",
            testo,
            flags=re.IGNORECASE | re.DOTALL
        )
        if nome.strip()
    ]

    conteggi_annotazioni[len(nomi)] += 1
    frequenze_nomi.update(nomi)

    if len(nomi) > 0:
        esempi.setdefault(len(nomi), (path, testo, nomi))

print(f"Email analizzate: {len(files)}")

print("\n--- Numero di annotazioni per email ---")
for numero, frequenza in sorted(conteggi_annotazioni.items()):
    print(f"{numero} annotazioni: {frequenza} email")

print("\n--- Nomi annotati più frequenti ---")
for nome, frequenza in frequenze_nomi.most_common(20):
    print(f"{nome}: {frequenza}")

print("\n--- Esempi di email annotate ---")
for numero in sorted(esempi)[:5]:
    path, testo, nomi = esempi[numero]

    print(f"\nFile: {path.name}")
    print(f"Nomi annotati: {nomi}")
    print("Inizio del testo:")
    print(testo[:900].replace("\n", " "))
