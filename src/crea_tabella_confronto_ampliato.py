
from pathlib import Path
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

FILE_TFIDF_ORIG = DATA_DIR / "ranking_tfidf_ampliato_originale.csv"
FILE_TFIDF_ANON = DATA_DIR / "ranking_tfidf_ampliato_anonimizzato.csv"
FILE_LLM_ORIG = DATA_DIR / "ranking_llm_ampliato_originale.csv"
FILE_LLM_ANON = DATA_DIR / "ranking_llm_ampliato_anonimizzato.csv"

OUTPUT = DATA_DIR / "confronto_metodi_ampliato.csv"


def leggi_csv(percorso):
    with percorso.open("r", encoding="utf-8-sig", newline="") as f:
        return {
            r["document_id"]: r
            for r in csv.DictReader(f)
        }


tfidf_orig = leggi_csv(FILE_TFIDF_ORIG)
tfidf_anon = leggi_csv(FILE_TFIDF_ANON)
llm_orig = leggi_csv(FILE_LLM_ORIG)
llm_anon = leggi_csv(FILE_LLM_ANON)

insiemi = [
    set(tfidf_orig),
    set(tfidf_anon),
    set(llm_orig),
    set(llm_anon),
]

if not all(ids == insiemi[0] for ids in insiemi):
    raise ValueError(
        "I quattro file non contengono lo stesso insieme di email."
    )

risultati = []

for document_id in sorted(insiemi[0]):
    a = tfidf_orig[document_id]
    b = tfidf_anon[document_id]
    c = llm_orig[document_id]
    d = llm_anon[document_id]

    identita = a["identita_vera"]

    if any(
        r["identita_vera"] != identita
        for r in (b, c, d)
    ):
        raise ValueError(
            f"Identità reale incoerente per {document_id}"
        )

    risultati.append({
        "document_id": document_id,
        "identita_vera": identita,
        "tfidf_originale_posizione": int(a["posizione_vera"]),
        "tfidf_anonimizzato_posizione": int(b["posizione_vera"]),
        "llm_originale_posizione": int(c["posizione_vera"]),
        "llm_anonimizzato_posizione": int(d["posizione_vera"]),
        "tfidf_originale_top1": int(a["top1"]),
        "tfidf_anonimizzato_top1": int(b["top1"]),
        "llm_originale_top1": int(c["top1"]),
        "llm_anonimizzato_top1": int(d["top1"]),
        "tfidf_anonimizzato_primo": b["candidato_1"],
        "llm_anonimizzato_primo": d["primo_candidato"],
    })

campi = list(risultati[0].keys())

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=campi)
    writer.writeheader()
    writer.writerows(risultati)

print(f"Email confrontate: {len(risultati)}")
print(f"Tabella salvata in: {OUTPUT}")

print("\n--- TOP-1 PER METODO E CONDIZIONE ---")

metriche = [
    ("TF-IDF originale", "tfidf_originale_top1"),
    ("TF-IDF anonimizzato", "tfidf_anonimizzato_top1"),
    ("LLM originale", "llm_originale_top1"),
    ("LLM anonimizzato", "llm_anonimizzato_top1"),
]

for nome, campo in metriche:
    corretti = sum(r[campo] for r in risultati)
    totale = len(risultati)
    print(f"{nome}: {corretti}/{totale} = {corretti / totale:.1%}")

print("\n--- EMAIL IDENTIFICATE DOPO L'ANONIMIZZAZIONE ---")

for r in risultati:
    if r["llm_anonimizzato_top1"] or r["tfidf_anonimizzato_top1"]:
        print(
            f"{r['document_id']} | vero: {r['identita_vera']} | "
            f"TF-IDF: {r['tfidf_anonimizzato_posizione']}° | "
            f"LLM: {r['llm_anonimizzato_posizione']}°"
        )
