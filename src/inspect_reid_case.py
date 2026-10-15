
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "openai_results_100.jsonl"

TARGET_NAME = "Justin Boyd"
TARGET_DOCS = {
    "derrick-j_all_documents_50",
    "haedicke-m_all_documents_5430",
    "taylor-m_all_documents_4367",
}

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        if record.get("document_id") not in TARGET_DOCS:
            continue

        original = record.get("text_original", "")

        print("\n" + "=" * 70)
        print("Documento:", record["document_id"])

        # Cerchiamo sia il nome completo sia il nome proprio.
        for term in ["Justin Boyd", "Justin"]:
            start = 0
            count = 0

            while True:
                pos = original.casefold().find(term.casefold(), start)
                if pos == -1:
                    break

                count += 1
                left = max(0, pos - 180)
                right = min(len(original), pos + len(term) + 180)
                context = original[left:right].replace("\n", " ")

                print(f"\n[{term}] occorrenza {count}:")
                print(context)

                start = pos + len(term)

        print("\nTotale occorrenze Justin Boyd:",
              original.casefold().count("justin boyd"))
        print("Totale occorrenze Justin:",
              original.casefold().count("justin"))
