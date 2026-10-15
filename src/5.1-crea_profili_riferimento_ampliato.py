from pathlib import Path
import csv
import json
import re
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

MANIFEST = DATA_DIR / "email_manifest.csv"
VERIFICA = DATA_DIR / "verifica_identita_mittenti.csv"
DATASET = DATA_DIR / "dataset_680_baseline.jsonl"
OUTPUT = DATA_DIR / "profili_riferimento_ampliato.csv"


def normalizza(testo):
    return re.sub(r"[^a-z0-9]", "", testo.casefold())


# 1. Seleziona i candidati sulla base delle email TRAIN.
# Ogni candidato deve avere almeno 2 email TRAIN
# in cui il nome annotato corrisponde al mittente.
candidati = {}

with VERIFICA.open("r", encoding="utf-8-sig", newline="") as f:
    for riga in csv.DictReader(f):
        nome = riga["nome"].strip()

        if int(riga["email_train"]) >= 1:
            chiave = normalizza(nome)

            if chiave:
                candidati[chiave] = nome


# 2. Legge il manifest per associare document_id e mittente.
# Il manifest viene usato per le informazioni di identità,
# mentre i documenti ammessi sono verificati sul dataset baseline.
documenti_manifest = {}

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    for doc in csv.DictReader(f):
        document_id = doc["document_id"].strip()

        if document_id in documenti_manifest:
            raise ValueError(
                f"document_id duplicato nel manifest: {document_id}"
            )

        documenti_manifest[document_id] = doc


# 3. Legge il dataset baseline e seleziona esclusivamente
# i record TRAIN appartenenti ai candidati.
# I testi di riferimento saranno recuperati dal dataset
# nel passaggio successivo.
profili = defaultdict(list)
documenti_visti = set()

with DATASET.open("r", encoding="utf-8") as f:
    for numero_riga, linea in enumerate(f, start=1):
        if not linea.strip():
            continue

        record = json.loads(linea)

        document_id = record["document_id"]

        if document_id in documenti_visti:
            raise ValueError(
                f"document_id duplicato nel dataset: {document_id}"
            )

        documenti_visti.add(document_id)

        if record["split"].strip().casefold() != "train":
            continue

        if document_id not in documenti_manifest:
            raise ValueError(
                f"Documento presente nel dataset ma assente "
                f"nel manifest: {document_id}"
            )

        doc = documenti_manifest[document_id]
        mittente = doc["from_email"].strip().casefold()

        if "@" not in mittente:
            continue

        local_part = mittente.split("@", 1)[0]
        chiave = normalizza(local_part)

        if chiave not in candidati:
            continue

        profili[candidati[chiave]].append({
            "document_id": document_id,
            "relative_path": doc["relative_path"],
            "email_mittente": mittente,
        })


# 4. Scrive il CSV mantenendo lo schema già utilizzato.
righe = []

for candidato, documenti in profili.items():
    for doc in documenti:
        righe.append({
            "candidato": candidato,
            **doc,
        })

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    campi = [
        "candidato",
        "document_id",
        "relative_path",
        "email_mittente",
    ]

    writer = csv.DictWriter(f, fieldnames=campi)
    writer.writeheader()
    writer.writerows(righe)


# 5. Controlli e riepilogo.
chiavi_documenti = [
    (r["candidato"], r["document_id"])
    for r in righe
]

if len(chiavi_documenti) != len(set(chiavi_documenti)):
    raise ValueError("Errore: documenti duplicati nei profili.")

if any(
    documenti_manifest[r["document_id"]]["split"].strip().casefold()
    != "train"
    for r in righe
):
    raise ValueError("Errore: trovato un documento non TRAIN.")

print(f"Candidati con almeno un riferimento: {len(profili)}")
print(f"Email di riferimento TRAIN: {len(righe)}")
print(f"File creato: {OUTPUT}")

print("\n--- PROFILI DI RIFERIMENTO ---")

for candidato, documenti in sorted(profili.items()):
    print(f"{candidato}: {len(documenti)} email TRAIN")

print("\nControlli completati.")
