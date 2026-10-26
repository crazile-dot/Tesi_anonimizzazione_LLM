'''
Lo script analizza i risultati dell'anonimizzazione OpenAI contenuti in openai_results_100.jsonl 
per studiare la capacità del modello di riconoscere le entità PERSON al variare della loro 
presenza nei documenti. Per ogni email calcola il numero di menzioni annotate, il numero di 
entità uniche, i true positive, i false negative e l'entity-level recall. I documenti vengono 
raggruppati in quattro fasce in base al numero di menzioni PERSON (0, 1–2, 3–10 e oltre 10), 
per le quali vengono calcolati micro recall e macro recall. Lo script individua inoltre i documenti 
con recall più basso e quelli con il maggior numero di entità uniche non riconosciute. L'analisi 
consente di approfondire gli errori di M4 e verificare eventuali differenze prestazionali associate 
alla densità di entità nei testi.
'''
import json
from pathlib import Path
from statistics import mean

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "processed" / "openai_results_100.jsonl"


def normalize(text):
    return " ".join(text.strip().lower().split())


def get_recall_data(record):
    # PERSON uniche: usate per entity-level Recall
    gold_unique = {
        normalize(entity["text"])
        for entity in record["ground_truth"]
    }

    predicted_unique = {
        normalize(entity["original"])
        for entity in record["entities_detected"]
    }

    tp = len(gold_unique & predicted_unique)
    fn = len(gold_unique - predicted_unique)

    recall = tp / len(gold_unique) if gold_unique else None

    # Numero totale di mention PERSON
    gold_mentions = len(record["ground_truth"])

    return gold_mentions, len(gold_unique), tp, fn, recall


def get_band(n):
    if n == 0:
        return "0"
    elif n <= 2:
        return "1–2"
    elif n <= 10:
        return "3–10"
    else:
        return ">10"


# ============================================================
# LETTURA
# ============================================================

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))


# ============================================================
# ANALISI PER DOCUMENTO
# ============================================================

documents = []

for record in records:

    (
        gold_mentions,
        gold_unique,
        tp,
        fn,
        recall
    ) = get_recall_data(record)

    documents.append({
        "document_id": record["document_id"],
        "gold_mentions": gold_mentions,
        "gold_unique": gold_unique,
        "tp": tp,
        "fn": fn,
        "recall": recall,
        "band": get_band(gold_mentions)
    })


# ============================================================
# ANALISI PER FASCIA
# ============================================================

bands = ["0", "1–2", "3–10", ">10"]

print("=" * 90)
print("ANALISI ENTITY-LEVEL RECALL M4")
print("Fasce basate sul numero di mention PERSON nel documento")
print("=" * 90)

print()

for band in bands:

    docs = [d for d in documents if d["band"] == band]
    docs_with_gold = [d for d in docs if d["gold_unique"] > 0]

    total_mentions = sum(d["gold_mentions"] for d in docs)
    total_unique = sum(d["gold_unique"] for d in docs)
    total_tp = sum(d["tp"] for d in docs)
    total_fn = sum(d["fn"] for d in docs)

    micro_recall = (
        total_tp / (total_tp + total_fn)
        if total_unique > 0
        else None
    )

    macro_recall = (
        mean(d["recall"] for d in docs_with_gold)
        if docs_with_gold
        else None
    )

    mean_mentions = (
        mean(d["gold_mentions"] for d in docs)
        if docs
        else None
    )

    print(f"FASCIA: {band}")
    print(f"  Documenti                  : {len(docs)}")
    print(f"  Documenti con PERSON       : {len(docs_with_gold)}")
    print(f"  PERSON mention totali      : {total_mentions}")
    print(f"  PERSON uniche totali       : {total_unique}")
    print(f"  TP                         : {total_tp}")
    print(f"  FN                         : {total_fn}")
    print(f"  Mention medie/documento   : {mean_mentions:.2f}")

    if micro_recall is not None:
        print(f"  Micro Recall entity-level : {micro_recall:.4f}")

    if macro_recall is not None:
        print(f"  Macro Recall entity-level : {macro_recall:.4f}")

    print()


# ============================================================
# DOCUMENTI CON RECALL PIÙ BASSO
# ============================================================

print("=" * 90)
print("DOCUMENTI CON ENTITY-LEVEL RECALL PIÙ BASSO")
print("=" * 90)

valid_docs = [
    d for d in documents
    if d["gold_unique"] > 0
]

valid_docs.sort(key=lambda d: (d["recall"], -d["gold_mentions"]))

print()

for d in valid_docs[:15]:

    print(
        f"{d['document_id']:<40} "
        f"mention={d['gold_mentions']:>3} "
        f"unique={d['gold_unique']:>3} "
        f"TP={d['tp']:>3} "
        f"FN={d['fn']:>3} "
        f"recall={d['recall']:.4f} "
        f"fascia={d['band']}"
    )


# ============================================================
# DOCUMENTI CON PIÙ FN
# ============================================================

print()
print("=" * 90)
print("DOCUMENTI CON PIÙ PERSON UNICHE NON RICONOSCIUTE")
print("=" * 90)

by_fn = sorted(
    valid_docs,
    key=lambda d: (d["fn"], d["gold_unique"]),
    reverse=True
)

print()

for d in by_fn[:15]:

    print(
        f"{d['document_id']:<40} "
        f"mention={d['gold_mentions']:>3} "
        f"unique={d['gold_unique']:>3} "
        f"TP={d['tp']:>3} "
        f"FN={d['fn']:>3} "
        f"recall={d['recall']:.4f} "
        f"fascia={d['band']}"
    )