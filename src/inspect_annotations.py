from pathlib import Path
import re

PROJECT_PATH = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_PATH
    / "data"
    / "raw"
    / "enronRandom-XML"
)

# Prendiamo tutti i file, escludendo lo ZIP
files = [
    f for f in DATASET_PATH.rglob("*")
    if f.is_file() and f.suffix != ".zip"
]

print(f"Documenti trovati: {len(files)}\n")

# Salviamo un esempio per ogni categoria
examples = {
    "0": None,
    "1": None,
    "2+": None
}

pattern = r"<true_name>(.*?)</true_name>"

for file_path in files:

    text = file_path.read_text(
        encoding="latin-1",
        errors="replace"
    )

    matches = re.findall(
        pattern,
        text,
        flags=re.DOTALL
    )

    count = len(matches)

    if count == 0 and examples["0"] is None:
        examples["0"] = file_path

    elif count == 1 and examples["1"] is None:
        examples["1"] = file_path

    elif count >= 2 and examples["2+"] is None:
        examples["2+"] = file_path

    # Se abbiamo trovato tutti e tre i casi, possiamo fermarci
    if all(examples.values()):
        break


print("ESEMPI TROVATI\n")

for category, file_path in examples.items():

    print(f"--- DOCUMENTO CON {category} PERSON ---")

    if file_path:
        print(file_path)

        text = file_path.read_text(
            encoding="latin-1",
            errors="replace"
        )

        matches = re.findall(
            pattern,
            text,
            flags=re.DOTALL
        )

        print("PERSON trovate:")

        for match in matches:
            print("-", match.strip())

    else:
        print("Nessun esempio trovato")

    print()