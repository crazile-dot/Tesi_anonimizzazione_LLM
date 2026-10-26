
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURAZIONE
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

BASELINE_FILE = DATA / "dataset_680_baseline.jsonl"
M4_FILE = DATA / "openai_results_680_m4_v2.jsonl"
M5_FILE = DATA / "openai_results_680_m5.jsonl"
PROFILES_FILE = DATA / "profili_testuali_ampliati.csv"
MANIFEST_FILE = DATA / "email_manifest.csv"
CANDIDATES_FILE = DATA / "candidati_attacco.csv"

OUTPUT_DETAIL = DATA / "ranking_tfidf_tutte_strategie_dettaglio.csv"
OUTPUT_SUMMARY = DATA / "ranking_tfidf_tutte_strategie_riepilogo.csv"
OUTPUT_MAPPING = DATA / "ranking_tfidf_controllo_mittenti.csv"

TOP_K = (1, 3, 5, 10)


# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================

def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON non valido in {path.name}, "
                        f"riga {line_number}: {exc}"
                    ) from exc
    return records


def normalize(value):
    """Normalizzazione per confrontare nomi ed email."""
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def email_localpart(email):
    email = str(email or "").strip().lower()
    if "@" not in email:
        return ""
    return email.split("@", 1)[0]


def name_matches_email(candidate, email):
    """
    Match conservativo: confronta il nome candidato con la parte
    locale dell'email, considerando forme come 'nome.cognome'.
    Non pretende di risolvere automaticamente tutti gli alias.
    """
    name = normalize(candidate)
    local = normalize(email_localpart(email))

    if not name or not local:
        return False

    name_tokens = name.split()
    local_tokens = local.split()

    # Esempio: "phillip allen" <-> "phillip.allen"
    if name_tokens == local_tokens:
        return True

    # Consente iniziali nel nome dell'indirizzo:
    # "mike grigsby" <-> "m.grigsby"
    if len(name_tokens) == 2 and len(local_tokens) == 2:
        first, last = name_tokens
        email_first, email_last = local_tokens
        if last == email_last and len(email_first) == 1:
            return first.startswith(email_first)

    return False


def read_text_field(record, *keys):
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


# ============================================================
# CARICAMENTO DATI
# ============================================================

print("Caricamento dei file...")

baseline_records = read_jsonl(BASELINE_FILE)
m4_records = read_jsonl(M4_FILE)
m5_records = read_jsonl(M5_FILE)

manifest = pd.read_csv(MANIFEST_FILE).fillna("")
profiles = pd.read_csv(PROFILES_FILE).fillna("")
candidates = pd.read_csv(CANDIDATES_FILE).fillna("")

required_manifest = {"document_id", "split", "from_email"}
required_profiles = {"candidato", "profilo_testuale"}

if not required_manifest.issubset(manifest.columns):
    raise ValueError(
        f"Colonne mancanti in email_manifest.csv: "
        f"{required_manifest - set(manifest.columns)}"
    )

if not required_profiles.issubset(profiles.columns):
    raise ValueError(
        f"Colonne mancanti in profili_testuali_ampliati.csv: "
        f"{required_profiles - set(profiles.columns)}"
    )

profiles["candidate_norm"] = profiles["candidato"].map(normalize)
profiles = profiles[
    (profiles["candidate_norm"] != "")
    & (profiles["profilo_testuale"].astype(str).str.strip() != "")
].copy()

# Un solo profilo per identità normalizzata.
profiles = profiles.drop_duplicates("candidate_norm", keep="first")
candidate_names = profiles["candidato"].astype(str).tolist()
candidate_norms = profiles["candidate_norm"].tolist()
profile_texts = profiles["profilo_testuale"].astype(str).tolist()

if not profile_texts:
    raise ValueError("Non sono stati caricati profili candidati validi.")

# Ground truth per email: il mittente viene ricavato dal manifest.
manifest["document_id"] = manifest["document_id"].astype(str)
manifest["split"] = manifest["split"].astype(str).str.lower()
manifest["from_email"] = manifest["from_email"].astype(str).str.strip()

manifest_by_id = manifest.drop_duplicates("document_id").set_index("document_id")

# Candidati dichiarati nel CSV.
candidate_set = set(candidate_norms)

# Mappe document_id -> record per ciascuna strategia.
baseline_by_id = {
    str(r["document_id"]): r
    for r in baseline_records
    if r.get("document_id") is not None
}

m4_by_id = {
    str(r["document_id"]): r
    for r in m4_records
    if r.get("document_id") is not None
}

m5_by_id = {
    str(r["document_id"]): r
    for r in m5_records
    if r.get("document_id") is not None
}

# Strategia -> nome del campo testuale.
baseline_fields = {
    "M0": "M0_originale",
    "M1": "M1_redacted",
    "M2": "M2_persona",
    "M3": "M3_hmac",
}

strategies = ["M0", "M1", "M2", "M3", "M4", "M5"]


# ============================================================
# COSTRUZIONE DEL SET DI DOCUMENTI COMUNE
# ============================================================

common_ids = sorted(
    set(baseline_by_id)
    & set(m4_by_id)
    & set(m5_by_id)
    & set(manifest_by_id.index)
)

# Consideriamo esclusivamente il test set.
common_ids = [
    doc_id for doc_id in common_ids
    if manifest_by_id.loc[doc_id, "split"] == "test"
]

# Risoluzione prudente del mittente:
# accettiamo solo un'associazione univoca con un candidato.
document_rows = []
mapping_rows = []

for doc_id in common_ids:
    sender_email = str(manifest_by_id.loc[doc_id, "from_email"]).strip()

    matching_candidates = [
        candidate_names[i]
        for i in range(len(candidate_names))
        if name_matches_email(candidate_names[i], sender_email)
    ]

    # Non usare il testo della mail da classificare per assegnare
    # la ground truth: sarebbe una contaminazione della valutazione.
    if len(matching_candidates) == 1:
        true_candidate = matching_candidates[0]
        status = "match_univoco"
    elif len(matching_candidates) == 0:
        true_candidate = ""
        status = "nessun_match"
    else:
        true_candidate = ""
        status = "match_ambiguo"

    mapping_rows.append({
        "document_id": doc_id,
        "from_email": sender_email,
        "candidati_corrispondenti": " | ".join(matching_candidates),
        "numero_match": len(matching_candidates),
        "stato_mapping": status,
        "mittente_candidato": true_candidate,
    })

    if status != "match_univoco":
        continue

    base = baseline_by_id[doc_id]
    rec4 = m4_by_id[doc_id]
    rec5 = m5_by_id[doc_id]

    texts = {
        "M0": read_text_field(base, "M0_originale", "testo_originale"),
        "M1": read_text_field(base, "M1_redacted"),
        "M2": read_text_field(base, "M2_persona"),
        "M3": read_text_field(base, "M3_hmac"),
        "M4": read_text_field(rec4, "text_anonymized"),
        "M5": read_text_field(rec5, "text_anonymized"),
    }

    # Verifica che il documento sia disponibile per tutte le strategie.
    if any(not text.strip() for text in texts.values()):
        mapping_rows[-1]["stato_mapping"] = "testo_mancante"
        continue

    document_rows.append({
        "document_id": doc_id,
        "from_email": sender_email,
        "true_candidate": true_candidate,
        "texts": texts,
    })

pd.DataFrame(mapping_rows).to_csv(
    OUTPUT_MAPPING, index=False, encoding="utf-8-sig"
)

print(f"Documenti test con record completo: {len(common_ids)}")
print(f"Documenti con mittente associato univocamente: {len(document_rows)}")
print(f"Controllo mapping salvato in: {OUTPUT_MAPPING}")

if len(document_rows) == 0:
    raise RuntimeError(
        "Nessun mittente associato univocamente. "
        "Apri il CSV di controllo: il mapping email-nome va verificato "
        "prima di calcolare le metriche."
    )


# ============================================================
# RANKING TF-IDF
# ============================================================

# Un unico spazio TF-IDF condiviso da profili e testi di tutte le strategie.
# Questo rende il confronto più coerente tra M0-M5.
corpus = profile_texts[:]
for row in document_rows:
    corpus.extend(row["texts"][s] for s in strategies)

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=300_000,
    token_pattern=r"(?u)\b\w+\b",
)

print("Calcolo della matrice TF-IDF...")
matrix = vectorizer.fit_transform(corpus)

n_profiles = len(profile_texts)
profile_matrix = matrix[:n_profiles]

details = []
offset = n_profiles

for row in document_rows:
    for strategy in strategies:
        text = row["texts"][strategy]
        text_vector = matrix[offset]
        offset += 1

        scores = cosine_similarity(text_vector, profile_matrix).ravel()

        # Ordinamento deterministico in caso di punteggio identico.
        ranking_indices = sorted(
            range(len(candidate_names)),
            key=lambda i: (-float(scores[i]), candidate_norms[i])
        )

        ranked_names = [candidate_names[i] for i in ranking_indices]
        ranked_norms = [candidate_norms[i] for i in ranking_indices]
        ranked_scores = [float(scores[i]) for i in ranking_indices]

        true_norm = normalize(row["true_candidate"])

        rank = next(
            (i + 1 for i, name in enumerate(ranked_norms)
             if name == true_norm),
            None
        )

        result = {
            "document_id": row["document_id"],
            "strategia": strategy,
            "from_email": row["from_email"],
            "mittente_candidato": row["true_candidate"],
            "rank_corretto": rank,
            "reciprocal_rank": 1.0 / rank if rank else 0.0,
            "primo_candidato": ranked_names[0] if ranked_names else "",
            "primo_punteggio": ranked_scores[0] if ranked_scores else 0.0,
            "ranking_top10": " | ".join(ranked_names[:10]),
        }

        for k in TOP_K:
            result[f"Top{k}"] = int(rank is not None and rank <= k)

        details.append(result)

detail_df = pd.DataFrame(details)
detail_df.to_csv(OUTPUT_DETAIL, index=False, encoding="utf-8-sig")


# ============================================================
# RIEPILOGO METRICHE
# ============================================================

summary_rows = []

for strategy in strategies:
    subset = detail_df[detail_df["strategia"] == strategy]
    n = len(subset)

    summary = {
        "strategia": strategy,
        "n_documenti": n,
    }

    for k in TOP_K:
        hits = int(subset[f"Top{k}"].sum())
        summary[f"Top{k}_n"] = hits
        summary[f"Top{k}_percentuale"] = hits / n if n else 0.0

    summary["MRR"] = subset["reciprocal_rank"].mean() if n else 0.0
    summary_rows.append(summary)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

print("\n=== RIEPILOGO RANKING TF-IDF ===")
print(summary_df.to_string(index=False, formatters={
    col: (lambda x: f"{x:.4f}")
    for col in summary_df.select_dtypes(include="number").columns
    if "percentuale" in col or col == "MRR"
}))

print("\nFile creati:")
print(f"- Dettaglio: {OUTPUT_DETAIL}")
print(f"- Riepilogo: {OUTPUT_SUMMARY}")
print(f"- Controllo mittenti: {OUTPUT_MAPPING}")
