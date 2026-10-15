from pathlib import Path
import re
import json

# Percorso della cartella principale del progetto
PROJECT_PATH = Path(__file__).resolve().parent.parent

# File di test
TEST_FILE = (
    PROJECT_PATH
    / "data"
    / "raw"
    / "enronRandom-XML"
    / "train"
    / "enron0"
    / "arnold-j_all_documents_1027"
)


def parse_document(file_path: Path) -> dict:
    """
    Legge un documento Enron ed estrae:
    - testo pulito
    - entità PERSON annotate con <true_name>
    - posizione delle entità nel testo pulito
    """

    # Leggiamo il file
    raw_text = file_path.read_text(
        encoding="latin-1",
        errors="replace"
    )

    entities = []

    # Pattern per trovare <true_name>...</true_name>
    pattern = r"<true_name>(.*?)</true_name>"

    # Costruiamo progressivamente il testo pulito
    clean_text = ""
    last_end = 0

    for match in re.finditer(pattern, raw_text, flags=re.DOTALL):

        # Testo prima dell'entità
        clean_text += raw_text[last_end:match.start()]

        # Nome dell'entità senza spazi superflui
        entity_text = match.group(1).strip()

        # Posizione di inizio nel testo pulito
        start = len(clean_text)

        # Inseriamo il nome nel testo pulito
        clean_text += entity_text

        # Posizione finale
        end = len(clean_text)

        # Salviamo l'entità
        entities.append({
            "type": "PERSON",
            "text": entity_text,
            "start": start,
            "end": end
        })

        # Aggiorniamo la posizione
        last_end = match.end()

    # Aggiungiamo il resto del documento
    clean_text += raw_text[last_end:]

    return {
        "document_id": file_path.name,
        "text_original": clean_text,
        "entities": entities
    }


# Parsing del documento
document = parse_document(TEST_FILE)

# Stampiamo il risultato
print("\nDOCUMENT ID:")
print(document["document_id"])

print("\nENTITÀ TROVATE:")
for entity in document["entities"]:
    print(entity)

print("\nTESTO CONTESTUALE DELL'ENTITÀ:")

for entity in document["entities"]:
    start = max(0, entity["start"] - 50)
    end = min(len(document["text_original"]), entity["end"] + 50)

    print(
        document["text_original"][start:end]
    )

# Salviamo anche il risultato in JSON
OUTPUT_PATH = (
    PROJECT_PATH
    / "data"
    / "processed"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_PATH / "test_document.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        document,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"\nFile JSON salvato in:\n{output_file}")