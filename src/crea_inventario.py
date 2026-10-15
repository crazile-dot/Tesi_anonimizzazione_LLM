
from pathlib import Path
from collections import Counter
import csv
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "raw" / "enronRandom-XML"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "email_manifest.csv"

files = sorted(
    p for p in DATASET_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() != ".zip"
)

righe = []
frequenze_persone = Counter()

for path in files:
    testo = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    # Estrae le persone annotate, mantenendo l'ordine.
    nomi = [
        re.sub(r"\s+", " ", nome).strip()
        for nome in re.findall(
            r"<true_name>(.*?)</true_name>",
            testo,
            flags=re.IGNORECASE | re.DOTALL
        )
    ]

    nomi = [nome for nome in nomi if nome]

    # Estrae il mittente dall'intestazione, se presente.
    match = re.search(
        r"(?im)^From:\s*(.+?)\s*$",
        testo
    )
    mittente = match.group(1).strip() if match else ""

    # Conta ogni persona al massimo una volta per email.
    persone_uniche = sorted(set(nomi), key=str.casefold)
    frequenze_persone.update(persone_uniche)

    righe.append({
        "document_id": path.stem,
        "relative_path": str(path.relative_to(DATASET_DIR)),
        "split": path.relative_to(DATASET_DIR).parts[0],
        "from_email": mittente,
        "annotation_count": len(nomi),
        "unique_person_count": len(persone_uniche),
        "annotated_names": " | ".join(persone_uniche),
    })

campi = [
    "document_id",
    "relative_path",
    "split",
    "from_email",
    "annotation_count",
    "unique_person_count",
    "annotated_names",
]

with output_file.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=campi)
    writer.writeheader()
    writer.writerows(righe)

print(f"Email inventariate: {len(righe)}")
print(f"Email con almeno una persona annotata: "
      f"{sum(r['unique_person_count'] > 0 for r in righe)}")
print(f"Persone distinte, contando le varianti testuali: "
      f"{len(frequenze_persone)}")
print(f"\nInventario salvato in:\n{output_file}")

print("\n--- Persone annotate più frequentemente ---")
for nome, frequenza in frequenze_persone.most_common(15):
    print(f"{nome}: {frequenza} email")

print("\n--- Distribuzione per split ---")
for split in sorted({r["split"] for r in righe}):
    subset = [r for r in righe if r["split"] == split]
    print(f"{split}: {len(subset)} email")
