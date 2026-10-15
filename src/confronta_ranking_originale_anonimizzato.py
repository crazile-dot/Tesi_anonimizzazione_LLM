
from pathlib import Path
import json
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Percorsi dei file
PROJECT_DIR = Path(__file__).resolve().parent.parent

CORPUS_ORIGINALE = (
    PROJECT_DIR / "data" / "processed" / "corpus_test.jsonl"
)
CORPUS_ANONIMO = (
    PROJECT_DIR / "data" / "processed" / "corpus_test_anonimo.jsonl"
)
PROFILI = (
    PROJECT_DIR / "data" / "processed" / "profili_testuali.csv"
)
RANKING_PRECEDENTE = (
    PROJECT_DIR / "data" / "processed" / "ranking_tfidf.csv"
)
OUTPUT = (
    PROJECT_DIR / "data" / "processed"
    / "confronto_ranking_originale_anonimizzato.csv"
)


def leggi_jsonl(percorso):
    with percorso.open("r", encoding="utf-8") as f:
        return {
            r["document_id"]: r
            for r in (json.loads(line) for line in f if line.strip())
        }


def basename(percorso):
    return percorso.replace("\\", "/").rsplit("/", 1)[-1]


# Leggiamo i dati
originali = leggi_jsonl(CORPUS_ORIGINALE)
anonimi = leggi_jsonl(CORPUS_ANONIMO)

with PROFILI.open("r", encoding="utf-8-sig", newline="") as f:
    profili = list(csv.DictReader(f))

with RANKING_PRECEDENTE.open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    ranking_precedente = list(csv.DictReader(f))


# Ground truth: mittente reale del documento
with (
    PROJECT_DIR / "data" / "processed" / "email_manifest.csv"
).open("r", encoding="utf-8-sig", newline="") as f:
    manifest = list(csv.DictReader(f))

mittenti_test = {
    basename(r["relative_path"]): r["from_email"].strip().casefold()
    for r in manifest
    if r["split"].strip().casefold() == "test"
}

email_candidati = {
    "scott.neal@enron.com": "scott neal",
    "errol.mclaughlin@enron.com": "errol mclaughlin",
    "drew.fossum@enron.com": "drew fossum",
    "mike.mcconnell@enron.com": "mike mcconnell",
}


# Ordine dei candidati e profili di riferimento invariati
nomi_candidati = [r["candidato"] for r in profili]
testi_profili = [r["profilo_testuale"] for r in profili]

# Usiamo lo stesso metodo TF-IDF del ranking precedente.
# Il vocabolario viene costruito sui profili di riferimento,
# senza utilizzare i documenti di test per apprendere il vocabolario.
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
)

matrice_profili = vectorizer.fit_transform(testi_profili)


def classifica(testo):
    vettore = vectorizer.transform([testo])
    similarita = cosine_similarity(vettore, matrice_profili)[0]

    return sorted(
        zip(nomi_candidati, similarita),
        key=lambda elemento: elemento[1],
        reverse=True,
    )


# Analizziamo esclusivamente i sei documenti già valutati
risultati = []

for riga in ranking_precedente:
    document_id = riga["document_id"]

    doc_originale = originali.get(document_id)
    doc_anonimo = anonimi.get(document_id)

    if doc_originale is None or doc_anonimo is None:
        print(f"Documento non trovato: {document_id}")
        continue

    mittente = mittenti_test.get(basename(document_id), "")
    identita_vera = email_candidati.get(mittente)

    if identita_vera is None:
        print(f"Mittente non associato ai candidati: {document_id}")
        continue

    # Testo originale e testo anonimizzato
    testo_originale = doc_originale["testo_originale"]
    testo_anonimo = doc_anonimo["testo_anonimizzato"]

    classifica_originale = classifica(testo_originale)
    classifica_anonima = classifica(testo_anonimo)

    posizione_originale = next(
        i + 1
        for i, (nome, _) in enumerate(classifica_originale)
        if nome == identita_vera
    )

    posizione_anonima = next(
        i + 1
        for i, (nome, _) in enumerate(classifica_anonima)
        if nome == identita_vera
    )

    risultati.append({
        "document_id": document_id,
        "identita_vera": identita_vera,
        "posizione_originale": posizione_originale,
        "primo_candidato_originale": classifica_originale[0][0],
        "similarita_originale": round(
            float(classifica_originale[0][1]), 6
        ),
        "posizione_anonimizzata": posizione_anonima,
        "primo_candidato_anonimizzato": classifica_anonima[0][0],
        "similarita_anonimizzata": round(
            float(classifica_anonima[0][1]), 6
        ),
    })


# Salviamo il confronto
if risultati:
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=risultati[0].keys())
        writer.writeheader()
        writer.writerows(risultati)


# Metriche Top-1
n = len(risultati)

if n:
    top1_originale = sum(
        r["posizione_originale"] == 1 for r in risultati
    ) / n

    top1_anonimizzata = sum(
        r["posizione_anonimizzata"] == 1 for r in risultati
    ) / n

    print(f"Documenti confrontati: {n}")
    print(
        f"Top-1 originale: {top1_originale:.3f} "
        f"({sum(r['posizione_originale'] == 1 for r in risultati)}/{n})"
    )
    print(
        f"Top-1 anonimizzato: {top1_anonimizzata:.3f} "
        f"({sum(r['posizione_anonimizzata'] == 1 for r in risultati)}/{n})"
    )

    print("\nDettaglio per documento:")

    for r in risultati:
        print(
            f"{r['document_id']} | "
            f"vera: {r['identita_vera']} | "
            f"originale: posizione {r['posizione_originale']} "
            f"(primo: {r['primo_candidato_originale']}) | "
            f"anonimizzato: posizione {r['posizione_anonimizzata']} "
            f"(primo: {r['primo_candidato_anonimizzato']})"
        )

    print(f"\nConfronto salvato in: {OUTPUT}")
else:
    print("Nessun documento valutabile.")
