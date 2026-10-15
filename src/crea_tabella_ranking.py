
import pandas as pd
from pathlib import Path

# Cartelle del progetto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

# Lettura dei risultati
tfidf = pd.read_csv(DATA_DIR / "confronto_ranking_originale_anonimizzato.csv")
llm_anon = pd.read_csv(DATA_DIR / "ranking_llm.csv")
llm_orig = pd.read_csv(DATA_DIR / "ranking_llm_originale.csv")

# Selezione e rinomina delle colonne TF-IDF
tfidf = tfidf[
    [
        "document_id",
        "identita_vera",
        "posizione_originale",
        "primo_candidato_originale",
        "posizione_anonimizzata",
        "primo_candidato_anonimizzato",
    ]
].copy()

tfidf = tfidf.rename(
    columns={
        "posizione_originale": "tfidf_originale_posizione_vera",
        "primo_candidato_originale": "tfidf_originale_primo_candidato",
        "posizione_anonimizzata": "tfidf_anonimizzato_posizione_vera",
        "primo_candidato_anonimizzato": "tfidf_anonimizzato_primo_candidato",
    }
)

# Selezione delle colonne LLM
llm_anon = llm_anon[
    ["document_id", "posizione_vera", "primo_candidato", "top1_corretto"]
].copy()

llm_anon = llm_anon.rename(
    columns={
        "posizione_vera": "llm_anonimizzato_posizione_vera",
        "primo_candidato": "llm_anonimizzato_primo_candidato",
        "top1_corretto": "llm_anonimizzato_top1_corretto",
    }
)

llm_orig = llm_orig[
    ["document_id", "posizione_vera", "primo_candidato", "top1_corretto"]
].copy()

llm_orig = llm_orig.rename(
    columns={
        "posizione_vera": "llm_originale_posizione_vera",
        "primo_candidato": "llm_originale_primo_candidato",
        "top1_corretto": "llm_originale_top1_corretto",
    }
)

# Controllo che ogni documento compaia una sola volta in ogni file
for nome, df in [
    ("TF-IDF", tfidf),
    ("LLM anonimizzato", llm_anon),
    ("LLM originale", llm_orig),
]:
    if df["document_id"].duplicated().any():
        raise ValueError(f"Documenti duplicati nel file: {nome}")

# Unione dei quattro esperimenti
risultati = tfidf.merge(
    llm_anon, on="document_id", how="inner", validate="one_to_one"
)

risultati = risultati.merge(
    llm_orig, on="document_id", how="inner", validate="one_to_one"
)

# Verifica dell'identità reale
for col in ["identita_vera"]:
    risultati[col] = risultati[col].astype(str).str.lower().str.strip()

# Indicatori Top-1 per tutti i metodi
risultati["tfidf_originale_top1_corretto"] = (
    risultati["tfidf_originale_posizione_vera"] == 1
).astype(int)

risultati["tfidf_anonimizzato_top1_corretto"] = (
    risultati["tfidf_anonimizzato_posizione_vera"] == 1
).astype(int)

# Ordinamento leggibile
risultati = risultati.sort_values("document_id")

# Salvataggio della tabella
output = DATA_DIR / "tabella_confronto_metodi.csv"
risultati.to_csv(output, index=False, encoding="utf-8-sig")

# Riepilogo delle accuratezze
metodi = {
    "TF-IDF originale": "tfidf_originale_top1_corretto",
    "TF-IDF anonimizzato": "tfidf_anonimizzato_top1_corretto",
    "LLM originale": "llm_originale_top1_corretto",
    "LLM anonimizzato": "llm_anonimizzato_top1_corretto",
}

print(f"Documenti confrontati: {len(risultati)}")
print(f"Tabella salvata in: {output}")
print("\n--- TOP-1 ACCURACY ---")

for nome, colonna in metodi.items():
    corretti = int(risultati[colonna].sum())
    totale = len(risultati)
    accuracy = corretti / totale if totale else 0
    print(f"{nome}: {accuracy:.3f} ({corretti}/{totale})")

print("\n--- CONFRONTO PER DOCUMENTO ---")
print(
    risultati[
        [
            "document_id",
            "identita_vera",
            "tfidf_originale_posizione_vera",
            "tfidf_anonimizzato_posizione_vera",
            "llm_originale_posizione_vera",
            "llm_anonimizzato_posizione_vera",
        ]
    ].to_string(index=False)
)
