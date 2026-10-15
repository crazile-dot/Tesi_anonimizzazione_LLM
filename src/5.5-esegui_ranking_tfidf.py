
from pathlib import Path
import csv
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PERCORSI
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

INPUT_TEST = DATA_DIR / "attack_test.jsonl"
INPUT_PROFILI = DATA_DIR / "profili_testuali_ampliati.csv"
INPUT_CANDIDATI = DATA_DIR / "candidati_attacco.csv"
INPUT_MANIFEST = DATA_DIR / "email_manifest.csv"

OUTPUT_SUMMARY = DATA_DIR / "riepilogo_ranking_tfidf.csv"


# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================

def normalizza(testo):
    """Rimuove spazi e caratteri non alfanumerici."""
    return re.sub(r"[^a-z0-9]", "", str(testo).casefold())


def leggi_jsonl(percorso):
    records = []
    with percorso.open("r", encoding="utf-8") as f:
        for numero_riga, riga in enumerate(f, start=1):
            if not riga.strip():
                continue
            try:
                records.append(json.loads(riga))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON non valido in {percorso}, riga {numero_riga}"
                ) from exc
    return records


def leggi_csv(percorso):
    with percorso.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def local_part(email):
    """Estrae la parte dell'indirizzo prima della @."""
    return str(email or "").strip().casefold().split("@")[0]


def scrivi_csv(percorso, righe, colonne):
    with percorso.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=colonne)
        writer.writeheader()
        writer.writerows(righe)


# ============================================================
# 1. CARICAMENTO DEI CANDIDATI E DEI PROFILI TRAIN
# ============================================================

righe_profili = leggi_csv(INPUT_PROFILI)
righe_candidati = leggi_csv(INPUT_CANDIDATI)

profili = {}
for riga in righe_profili:
    nome = riga["candidato"].strip()
    testo = riga["profilo_testuale"].strip()

    if nome and testo:
        chiave = normalizza(nome)

        if chiave in profili:
            raise ValueError(
                f"Nome candidato duplicato dopo la normalizzazione: {nome}"
            )

        profili[chiave] = {
            "nome": nome,
            "testo": testo,
        }

# Il file dei candidati stabilisce il catalogo dell'attacco.
candidati_csv = {
    normalizza(riga["nome"]): riga["nome"].strip()
    for riga in righe_candidati
    if riga.get("nome", "").strip()
}

# Manteniamo soltanto i candidati per cui esiste un profilo testuale.
chiavi_candidati = [
    chiave for chiave in candidati_csv
    if chiave in profili
]

if len(chiavi_candidati) < 2:
    raise ValueError(
        "Numero di candidati insufficiente: controlla i file di input."
    )

nomi_candidati = [profili[k]["nome"] for k in chiavi_candidati]
testi_profili = [profili[k]["testo"] for k in chiavi_candidati]

print(f"Candidati con profilo utilizzabile: {len(chiavi_candidati)}")


# ============================================================
# 2. COSTRUZIONE DELLO SPAZIO TF-IDF SUI SOLI PROFILI TRAIN
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
)

matrice_profili = vectorizer.fit_transform(testi_profili)


# ============================================================
# 3. CARICAMENTO TEST E GROUND TRUTH
# ============================================================

test_records = leggi_jsonl(INPUT_TEST)
manifest = leggi_csv(INPUT_MANIFEST)

mittente_per_documento = {
    riga["document_id"]: riga.get("from_email", "")
    for riga in manifest
}

# La ground truth è il mittente del documento, non una persona
# semplicemente menzionata nel corpo dell'email.
candidato_per_chiave = {
    normalizza(profili[k]["nome"]): profili[k]["nome"]
    for k in chiavi_candidati
}

test_utilizzabili = []
esclusi = 0

for record in test_records:
    document_id = record["document_id"]
    mittente = mittente_per_documento.get(document_id, "")
    chiave_mittente = normalizza(local_part(mittente))

    identita_vera = candidato_per_chiave.get(chiave_mittente)

    # Escludiamo i documenti senza corrispondenza esatta
    # tra mittente e candidato, senza inventare un'etichetta.
    if not identita_vera:
        esclusi += 1
        continue

    test_utilizzabili.append({
        "document_id": document_id,
        "identita_vera": identita_vera,
        "record": record,
    })

print(f"Email presenti in attack_test.jsonl: {len(test_records)}")
print(f"Email TEST valutabili: {len(test_utilizzabili)}")
print(f"Email TEST escluse per mittente non associato: {esclusi}")

if not test_utilizzabili:
    raise ValueError(
        "Nessuna email TEST associabile ai candidati. "
        "Controlla document_id, from_email e nomi dei candidati."
    )


# ============================================================
# 4. RANKING DELLE QUATTRO CONDIZIONI
# ============================================================

condizioni = {
    "Originale": "testo_originale",
    "M1": "M1_redacted",
    "M2": "M2_persona",
    "M3": "M3_hmac",
}

riepilogo = []

for nome_condizione, campo_testo in condizioni.items():
    righe_ranking = []
    posizioni_corrette = []

    for elemento in test_utilizzabili:
        record = elemento["record"]
        testo_query = str(record.get(campo_testo, "") or "").strip()

        if not testo_query:
            print(
                f"ATTENZIONE: testo vuoto per "
                f"{elemento['document_id']} nella condizione "
                f"{nome_condizione}"
            )
            continue

        vettore_query = vectorizer.transform([testo_query])
        similarita = cosine_similarity(
            vettore_query,
            matrice_profili
        ).ravel()

        ordine = similarita.argsort()[::-1]
        classifica = [
            {
                "candidato": nomi_candidati[i],
                "score": float(similarita[i]),
            }
            for i in ordine
        ]

        identita_vera = elemento["identita_vera"]
        posizione_vera = next(
            (
                posizione
                for posizione, voce in enumerate(classifica, start=1)
                if normalizza(voce["candidato"])
                == normalizza(identita_vera)
            ),
            None,
        )

        if posizione_vera is None:
            # Non dovrebbe accadere se la ground truth è nel catalogo.
            raise ValueError(
                f"Ground truth non presente nel ranking: {identita_vera}"
            )

        posizioni_corrette.append(posizione_vera)

        for posizione, voce in enumerate(classifica, start=1):
            righe_ranking.append({
                "document_id": elemento["document_id"],
                "condizione": nome_condizione,
                "identita_vera": identita_vera,
                "candidato": voce["candidato"],
                "score_tfidf": round(voce["score"], 8),
                "posizione": posizione,
                "candidato_corretto": (
                    normalizza(voce["candidato"])
                    == normalizza(identita_vera)
                ),
                "top1": posizione_vera == 1,
                "top3": posizione_vera <= 3,
                "top5": posizione_vera <= 5,
                "posizione_vera": posizione_vera,
            })

    nome_file = f"ranking_tfidf_ampliato_{nome_condizione.lower()}.csv"
    output = DATA_DIR / nome_file

    colonne_ranking = [
        "document_id",
        "condizione",
        "identita_vera",
        "candidato",
        "score_tfidf",
        "posizione",
        "candidato_corretto",
        "top1",
        "top3",
        "top5",
        "posizione_vera",
    ]

    scrivi_csv(output, righe_ranking, colonne_ranking)

    n = len(posizioni_corrette)
    top1 = sum(p <= 1 for p in posizioni_corrette) / n
    top3 = sum(p <= 3 for p in posizioni_corrette) / n
    top5 = sum(p <= 5 for p in posizioni_corrette) / n
    posizione_media = sum(posizioni_corrette) / n

    riepilogo.append({
        "condizione": nome_condizione,
        "email_valutate": n,
        "candidati": len(chiavi_candidati),
        "Top1": round(top1, 4),
        "Top3": round(top3, 4),
        "Top5": round(top5, 4),
        "posizione_media_vera": round(posizione_media, 2),
    })

    print(f"\nCondizione: {nome_condizione}")
    print(f"  Email valutate: {n}")
    print(f"  Top-1: {top1:.2%}")
    print(f"  Top-3: {top3:.2%}")
    print(f"  Top-5: {top5:.2%}")
    print(f"  Posizione media del mittente: {posizione_media:.2f}")
    print(f"  File creato: {output}")


# ============================================================
# 5. RIEPILOGO COMPARATIVO
# ============================================================

scrivi_csv(
    OUTPUT_SUMMARY,
    riepilogo,
    [
        "condizione",
        "email_valutate",
        "candidati",
        "Top1",
        "Top3",
        "Top5",
        "posizione_media_vera",
    ],
)

print("\nRIEPILOGO FINALE")
for riga in riepilogo:
    print(
        f"{riga['condizione']:10s} | "
        f"Top-1: {riga['Top1']:.2%} | "
        f"Top-3: {riga['Top3']:.2%} | "
        f"Top-5: {riga['Top5']:.2%} | "
        f"posizione media: {riga['posizione_media_vera']:.2f}"
    )

print(f"\nRiepilogo salvato in: {OUTPUT_SUMMARY}")
