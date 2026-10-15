'''
Lo script analizza i risultati dell'anonimizzazione OpenAI contenuti in openai_results_100.jsonl 
e confronta, per ciascun documento, le entità PERSON annotate nel ground truth con quelle 
rilevate dal modello. Dopo una normalizzazione dei nomi, individua le entità presenti nel 
riferimento ma assenti tra le predizioni, classificandole come false negative a livello di 
entità. Per ogni errore stampa il nome non riconosciuto, l'identificativo del documento, il
 numero di entità uniche annotate e la lunghezza del testo originale. Lo script supporta 
 l'analisi qualitativa degli errori di M4, ma non valuta direttamente la correttezza delle 
 sostituzioni nel testo anonimizzato.
'''
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "processed" / "openai_results_100.jsonl"


def normalize(text):
    return " ".join(text.strip().lower().split())


records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))


false_negatives = []


for record in records:

    gold = {
        normalize(entity["text"]): entity["text"]
        for entity in record["ground_truth"]
    }

    predicted = {
        normalize(entity["original"])
        for entity in record["entities_detected"]
    }

    missed = set(gold.keys()) - predicted

    for entity_norm in missed:

        false_negatives.append({
            "document_id": record["document_id"],
            "entity": gold[entity_norm],
            "document_length": len(record["text_original"]),
            "gold_entities": len(gold),
        })


# ============================================================
# RISULTATI
# ============================================================

print("=" * 100)
print("PERSON NON RICONOSCIUTE DA M4 (FALSE NEGATIVES)")
print("=" * 100)

print(f"\nNumero totale di FN: {len(false_negatives)}\n")


for i, fn in enumerate(false_negatives, start=1):

    print(
        f"{i:>3}. "
        f"{fn['entity']!r:<35} "
        f"doc={fn['document_id']:<40} "
        f"gold_unique={fn['gold_entities']:>3} "
        f"length={fn['document_length']:>5}"
    )