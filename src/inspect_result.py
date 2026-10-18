
import json
from pathlib import Path

doc_id = "davis-d_all_documents_440"

files = [
    Path("data/processed/openai_results_680_m4_v2.jsonl"),
    Path("data/processed/openai_results_680_m5.jsonl"),
]

for file_path in files:
    print("\n" + "=" * 80)
    print(f"FILE: {file_path}")
    print("=" * 80)

    if not file_path.exists():
        print("FILE NON TROVATO")
        continue

    found = False

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            # Cerca l'identificativo del documento in tutti i campi,
            # perché potrebbe non essere memorizzato in document_id.
            if doc_id in json.dumps(record, ensure_ascii=False):
                found = True
                print(json.dumps(
                    record,
                    ensure_ascii=False,
                    indent=2
                )[:15000])
                break

    if not found:
        print(f"Documento {doc_id} non trovato.")
