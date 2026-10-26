from pathlib import Path
import csv
import json
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

DATASET = DATA_DIR / "dataset_680_baseline.jsonl"
RIFERIMENTI = DATA_DIR / "profili_riferimento_ampliato.csv"
OUTPUT = DATA_DIR / "profili_testuali_ampliati.csv"


def rimuovi_tag_true_name(testo):
    """
    Rimuove i tag XML delle annotazioni, mantenendo
    il nome originale contenuto al loro interno.
    """
    return re.sub(
        r"<true_name>(.*?)</true_name>",
        r"\1",
        testo,
        flags=re.I | re.S,
    )


# 1. Carica i documenti TRAIN dal dataset baseline.
documenti_train = {}

with DATASET.open("r", encoding="utf-8") as f:
    for numero_riga, linea in enumerate(f, start=1):
        if not linea.strip():
            continue

        record = json.loads(linea)

        if record["split"].strip().casefold() != "train":
            continue

        document_id = record["document_id"]

        if document_id in documenti_train:
            raise ValueError(
                f"Documento TRAIN duplicato: {document_id}"
            )

        documenti_train[document_id] = record


# 2. Legge i riferimenti selezionati dal primo script.
with RIFERIMENTI.open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    riferimenti = list(csv.DictReader(f))


# 3. Costruisce un profilo testuale per ciascun candidato,
# recuperando esclusivamente i testi originali dei documenti TRAIN.
profili = {}
documenti_usati = set()

for riga in riferimenti:
    candidato = riga["candidato"]
    document_id = riga["document_id"]

    if document_id in documenti_usati:
        raise ValueError(
            f"Documento usato più volte nei riferimenti: {document_id}"
        )

    documenti_usati.add(document_id)

    if document_id not in documenti_train:
        raise ValueError(
            f"Il riferimento {document_id} non è presente "
            "tra i documenti TRAIN del dataset baseline."
        )

    record = documenti_train[document_id]
    testo = record["testo_originale"]

    if not testo or not testo.strip():
        print(f"ATTENZIONE: testo vuoto per {document_id}")
        continue

    testo = rimuovi_tag_true_name(testo)

    profili.setdefault(candidato, []).append(testo)


# 4. Scrive il CSV mantenendo lo schema atteso dagli script successivi.
with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    campi = ["candidato", "numero_email", "profilo_testuale"]
    writer = csv.DictWriter(f, fieldnames=campi)
    writer.writeheader()

    for candidato, testi in sorted(profili.items()):
        writer.writerow({
            "candidato": candidato,
            "numero_email": len(testi),
            "profilo_testuale": "\n\n".join(testi),
        })


# 5. Riepilogo finale.
print(f"Documenti TRAIN disponibili: {len(documenti_train)}")
print(f"Riferimenti letti: {len(riferimenti)}")
print(f"Profili testuali creati: {len(profili)}")
print(f"File creato: {OUTPUT}")

print("\n--- PROFILI TESTUALI ---")

for candidato, testi in sorted(profili.items()):
    print(
        f"{candidato}: {len(testi)} email, "
        f"{sum(len(t) for t in testi)} caratteri"
    )

print("\nControlli completati.")
