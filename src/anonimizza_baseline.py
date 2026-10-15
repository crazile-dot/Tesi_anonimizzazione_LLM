import json
from pathlib import Path


# Cartella principale del progetto
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset già anonimizzato con M1, M2 e M3
INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "dataset_680_baseline.jsonl"
)

# File di output per gli esperimenti
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

TRAIN_FILE = OUTPUT_DIR / "attack_train.jsonl"
TEST_FILE = OUTPUT_DIR / "attack_test.jsonl"


def main():
    # Verifica che il dataset di partenza esista
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_records = []
    test_records = []

    # Legge il dataset originale senza modificarlo
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON non valido alla riga {line_number}"
                ) from error

            # Controlla la presenza dello split originale
            split = record.get("split")

            if split == "train":
                train_records.append(record)

            elif split == "test":
                test_records.append(record)

            else:
                raise ValueError(
                    f"Split non valido alla riga {line_number}: "
                    f"{split!r}. Valori ammessi: 'train', 'test'."
                )

    # Scrive i record mantenendo tutti i campi originali
    for output_file, records in (
        (TRAIN_FILE, train_records),
        (TEST_FILE, test_records),
    ):
        with output_file.open(
            "w", encoding="utf-8", newline="\n"
        ) as file:
            for record in records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

    print("Suddivisione completata.")
    print(f"Input: {INPUT_FILE}")
    print(f"Train salvato in: {TRAIN_FILE}")
    print(f"Test salvato in:  {TEST_FILE}")
    print("Tutti i campi originali sono stati preservati.")


if __name__ == "__main__":
    main()
