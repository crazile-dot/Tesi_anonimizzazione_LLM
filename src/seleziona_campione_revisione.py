
import csv
import random
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"

INPUT_FILE = DATA / "registro_revisione_residui_m4_m5.csv"
OUTPUT_FILE = DATA / "campione_revisione_residui_m4_m5.csv"

SEED = 42
CAMPIONE_CONDIVISI = 20


def main():
    with INPUT_FILE.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    solo_m4 = [r for r in rows if r["categoria"] == "solo_M4"]
    solo_m5 = [r for r in rows if r["categoria"] == "solo_M5"]
    entrambi = [
        r for r in rows
        if r["categoria"] == "residuo_entrambi"
    ]

    rng = random.Random(SEED)
    n = min(CAMPIONE_CONDIVISI, len(entrambi))
    campione_entrambi = rng.sample(entrambi, n)

    selected = (
        solo_m4
        + solo_m5
        + campione_entrambi
    )

    # Aggiunge una colonna per distinguere la provenienza del caso.
    if "gruppo_revisione" not in fieldnames:
        fieldnames.append("gruppo_revisione")

    for row in solo_m4:
        row["gruppo_revisione"] = "solo_M4"

    for row in solo_m5:
        row["gruppo_revisione"] = "solo_M5"

    for row in campione_entrambi:
        row["gruppo_revisione"] = "residuo_entrambi"

    selected.sort(
        key=lambda r: (
            r["gruppo_revisione"],
            r["document_id"],
            r["annotazione_xml"],
        )
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Casi solo M4: {len(solo_m4)}")
    print(f"Casi solo M5: {len(solo_m5)}")
    print(f"Casi condivisi selezionati: {len(campione_entrambi)}")
    print(f"Totale campione: {len(selected)}")
    print(f"Seed casuale: {SEED}")
    print(f"File creato: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
