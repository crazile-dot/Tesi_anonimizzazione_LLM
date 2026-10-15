
from pathlib import Path
import json
import csv
import os

from openai import OpenAI


import re

def rimuovi_tag_xml(testo):
    return re.sub(r"</?true_name>", "", testo)



# Modello da utilizzare
MODEL_NAME = "gpt-6-luna"

# Percorsi del progetto
PROJECT_DIR = Path(__file__).resolve().parent.parent

CORPUS_ANONIMO = (
    PROJECT_DIR / "data" / "processed" / "corpus_test.jsonl"
)
PROFILI = (
    PROJECT_DIR / "data" / "processed" / "profili_testuali.csv"
)
RANKING_PRECEDENTE = (
    PROJECT_DIR / "data" / "processed" / "ranking_tfidf.csv"
)
OUTPUT = (
    PROJECT_DIR / "data" / "processed" / "ranking_llm_originale.csv"
)

# Il client legge OPENAI_API_KEY dalla variabile d'ambiente
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def leggi_jsonl(percorso):
    with percorso.open("r", encoding="utf-8") as f:
        return {
            r["document_id"]: r
            for r in (json.loads(line) for line in f if line.strip())
        }


# Caricamento dei dati
documenti = leggi_jsonl(CORPUS_ANONIMO)

with PROFILI.open("r", encoding="utf-8-sig", newline="") as f:
    profili = list(csv.DictReader(f))

with RANKING_PRECEDENTE.open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    documenti_da_valutare = list(csv.DictReader(f))

nomi_candidati = [r["candidato"] for r in profili]

risultati = []

for indice, riga in enumerate(documenti_da_valutare, start=1):
    document_id = riga["document_id"]
    identita_vera = riga["identita_vera"]

    documento = documenti.get(document_id)

    if documento is None:
        print(f"Documento non trovato: {document_id}")
        continue

    testo_email = rimuovi_tag_xml(documento["testo_originale"])

    # Prepariamo i profili dei candidati
    profili_testo = "\n\n".join(
        f"PROFILO CANDIDATO: {profilo['candidato']}\n"
        f"Email di riferimento:\n{profilo['profilo_testuale']}"
        for profilo in profili
    )

    prompt = f"""
Sei un analista che deve valutare un attacco di identificazione
dell'autore di un'email.

Sono disponibili quattro candidati e, per ciascuno, un profilo
costruito da email di riferimento.

Devi ordinare TUTTI i candidati dal più al meno compatibile
con l'email da identificare.

Usa il contenuto, gli argomenti, le abitudini linguistiche e
lo stile di scrittura. Non considerare il compito come una
semplice ricerca del nome: i nomi che compaiono nel corpo
possono riferirsi a persone diverse dall'autore.

Non inventare prove. Se gli elementi sono deboli, scegli
comunque un ordinamento basato sugli indizi disponibili.

CANDIDATI AMMESSI:
{json.dumps(nomi_candidati, ensure_ascii=False)}

PROFILI DI RIFERIMENTO:
{profili_testo}

EMAIL ANONIMIZZATA DA IDENTIFICARE:
{testo_email}

Restituisci esclusivamente un oggetto JSON valido nel formato:
{{
  "ranking": [
    {{"candidato": "nome esatto del candidato", "motivazione": "breve motivazione"}}
  ]
}}

La lista deve contenere ogni candidato ammesso esattamente una
volta, dal più al meno compatibile. Usa i nomi esattamente come
sono riportati nella lista dei candidati.
"""

    print(f"[{indice}/{len(documenti_da_valutare)}] {document_id}")

    risposta = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        reasoning={"effort": "low"},
        text={"format": {"type": "json_object"}},
    )

    dati = json.loads(risposta.output_text)
    ranking = dati["ranking"]

    nomi_ranking = [elemento["candidato"] for elemento in ranking]

    # Verifichiamo che il modello abbia classificato tutti i candidati
    if (
        len(nomi_ranking) != len(nomi_candidati)
        or set(nomi_ranking) != set(nomi_candidati)
        or len(set(nomi_ranking)) != len(nomi_ranking)
    ):
        raise ValueError(
            f"Ranking non valido per {document_id}: {nomi_ranking}"
        )

    posizione_vera = nomi_ranking.index(identita_vera) + 1

    risultati.append({
        "document_id": document_id,
        "identita_vera": identita_vera,
        "ranking_llm": " > ".join(nomi_ranking),
        "primo_candidato": nomi_ranking[0],
        "posizione_vera": posizione_vera,
        "top1_corretto": int(posizione_vera == 1),
        "motivazione_primo": ranking[0].get("motivazione", ""),
    })

    print(
        f"  Primo candidato: {nomi_ranking[0]} | "
        f"Posizione autore reale: {posizione_vera}"
    )


# Salvataggio dei risultati
if risultati:
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(risultati[0].keys())
        )
        writer.writeheader()
        writer.writerows(risultati)

    top1 = sum(r["top1_corretto"] for r in risultati) / len(risultati)

    print("\n--- RISULTATI LLM ---")
    print(f"Documenti valutati: {len(risultati)}")
    print(f"Top-1 accuracy: {top1:.3f} ({sum(r['top1_corretto'] for r in risultati)}/{len(risultati)})")
    print(f"Risultati salvati in: {OUTPUT}")
else:
    print("Nessun documento valutato.")
