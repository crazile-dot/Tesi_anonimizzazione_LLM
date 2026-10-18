
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "embeddings_680_all_v1.jsonl"
)

EXPECTED_METHODS = [
    "M0_original",
    "M1_redaction",
    "M2_pseudonymization",
    "M3_hmac",
    "M4_llm_substitution",
    "M5_llm_anonymization",
]

records = []

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print("Documenti:", len(records))
print("ID univoci:", len({r["document_id"] for r in records}))

for method in EXPECTED_METHODS:
    statuses = {}
    dimensions = set()
    invalid_vectors = 0
    chunk_counts = []

    for record in records:
        item = record["embeddings"].get(method)

        if item is None:
            status = "missing_method"
        else:
            status = item.get("status", "missing_status")

            vector = item.get("embedding")
            if vector is not None:
                array = np.asarray(vector, dtype=float)

                if (
                    array.ndim != 1
                    or not np.all(np.isfinite(array))
                ):
                    invalid_vectors += 1
                else:
                    dimensions.add(len(array))
                    norm = np.linalg.norm(array)

                    if not np.isclose(norm, 1.0, atol=1e-4):
                        invalid_vectors += 1

            chunk_counts.append(item.get("num_chunks", 0))

        statuses[status] = statuses.get(status, 0) + 1

    print(f"\n{method}")
    print("  Stati:", statuses)
    print("  Dimensioni vettori:", sorted(dimensions))
    print("  Vettori non validi o non normalizzati:", invalid_vectors)

    if chunk_counts:
        print("  Chunk min/max:", min(chunk_counts), max(chunk_counts))
