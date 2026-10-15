
from pathlib import Path
import re

# Cartella principale del progetto
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Cartella del dataset originale
DATASET_DIR = PROJECT_DIR / "data" / "raw" / "enronRandom-XML"

# Cerca i file contenuti nelle sottocartelle
files = [
    path
    for path in DATASET_DIR.rglob("*")
    if path.is_file()
]

print(f"Cartella dataset: {DATASET_DIR}")
print(f"File trovati: {len(files)}")

if not files:
    raise FileNotFoundError(
        f"Nessun file trovato in {DATASET_DIR}. "
        "Controlla il percorso del dataset."
    )

# Esamina alcuni file
print("\n--- Esempi di file ---")

for path in files[:5]:
    print(f"\nFile: {path.relative_to(DATASET_DIR)}")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"Errore di lettura: {exc}")
        continue

    # Cerca le annotazioni che identificano i nomi
    names = re.findall(
        r"<true_name>(.*?)</true_name>",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    print(f"Caratteri: {len(text)}")
    print(f"Annotazioni true_name: {len(names)}")
    print(f"Esempi di nomi annotati: {names[:3]}")

print("\nVerifica terminata.")
