
import csv
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"

INPUT_FILE = DATA / "audit_residui_m4_m5_680.csv"
OUTPUT_FILE = DATA / "registro_revisione_residui_m4_m5.csv"


def main():
    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Conserva soltanto i casi con almeno un residuo.
    residual_cases = [
        row for row in rows
        if (
            int(row["occorrenze_residue_m4"]) > 0
            or int(row["occorrenze_residue_m5"]) > 0
        )
    ]

    extra_fields = [
        "classificazione_m4",
        "classificazione_m5",
        "valutazione_manualе",
        "note_revisione",
    ]

    fieldnames = list(rows[0].keys()) + extra_fields if rows else []

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in residual_cases:
            for field in extra_fields:
                row[field] = ""
            writer.writerow(row)

    print(f"Casi totali nel CSV originale: {len(rows)}")
    print(f"Casi con residui in M4 o M5: {len(residual_cases)}")
    print(f"Registro salvato in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
