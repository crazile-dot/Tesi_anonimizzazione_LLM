import csv
from pathlib import Path

# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

FILE_M4 = DATA_DIR / "ranking_llm_ampliato_m4.csv"
FILE_M5 = DATA_DIR / "ranking_llm_ampliato_m5.csv"

OUTPUT_DETTAGLIO = DATA_DIR / "confronto_metriche_m4_m5_11_documenti.csv"
OUTPUT_RIEPILOGO = DATA_DIR / "riepilogo_confronto_m4_m5_11_documenti.txt"

# Il ranking contiene al massimo 10 candidati.
# 11 indica che l'identità vera non compare nella Top 10.


# ============================================================
# LETTURA E NORMALIZZAZIONE
# ============================================================

def leggi_csv(percorso):
    with percorso.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalizza_nome(nome):
    return " ".join((nome or "").casefold().split())


def calcola_posizione(riga):
    """
    Ricava la posizione dell'identità vera dal ranking.
    Se l'identità non compare tra i 10 candidati, restituisce 11.
    """

    identita = normalizza_nome(riga.get("identita_vera", ""))
    ranking = riga.get("ranking_llm", "") or ""

    candidati = [
        normalizza_nome(nome)
        for nome in ranking.split(">")
    ]

    if identita and candidati:
        for posizione, candidato in enumerate(candidati, start=1):
            if candidato == identita:
                return posizione

        # Ranking disponibile, identità non trovata.
        return 11

    # Fallback se il ranking testuale non è disponibile.
    valore = (riga.get("posizione_vera") or "").strip()

    if valore:
        try:
            posizione = int(float(valore))
            if 1 <= posizione <= 10:
                return posizione
        except ValueError:
            pass

    return 11


def reciprocal_rank(posizione):
    # Se l'identità non è nella Top 10, il contributo all'MRR è zero.
    return 1 / posizione if posizione <= 10 else 0.0


def calcola_metriche(posizioni):
    n = len(posizioni)

    if n == 0:
        raise ValueError("Nessun ranking disponibile.")

    metriche = {}

    for k in (1, 3, 5, 10):
        successi = sum(posizione <= k for posizione in posizioni)
        metriche[f"Top{k}"] = {
            "successi": successi,
            "percentuale": successi / n * 100,
        }

    metriche["MRR"] = (
        sum(reciprocal_rank(posizione) for posizione in posizioni) / n
    )

    return metriche


# ============================================================
# CARICAMENTO DEI RANKING
# ============================================================

for percorso in (FILE_M4, FILE_M5):
    if not percorso.exists():
        raise FileNotFoundError(f"File non trovato: {percorso}")

ranking_m4 = leggi_csv(FILE_M4)
ranking_m5 = leggi_csv(FILE_M5)

m4_per_id = {
    r["document_id"].strip(): r
    for r in ranking_m4
    if r.get("document_id", "").strip()
}

m5_per_id = {
    r["document_id"].strip(): r
    for r in ranking_m5
    if r.get("document_id", "").strip()
}

ids_comuni = sorted(set(m4_per_id) & set(m5_per_id))

if len(ids_comuni) != 11:
    raise ValueError(
        f"Attesi 11 documenti comuni, trovati {len(ids_comuni)}. "
        "Controlla i file di input."
    )


# ============================================================
# CONFRONTO DOCUMENTO PER DOCUMENTO
# ============================================================

righe_dettaglio = []
posizioni_m4 = []
posizioni_m5 = []

for document_id in ids_comuni:
    r4 = m4_per_id[document_id]
    r5 = m5_per_id[document_id]

    identita_m4 = r4.get("identita_vera", "").strip()
    identita_m5 = r5.get("identita_vera", "").strip()

    if (
        identita_m4
        and identita_m5
        and normalizza_nome(identita_m4) != normalizza_nome(identita_m5)
    ):
        raise ValueError(
            f"Identità vera discordante per {document_id}: "
            f"M4={identita_m4}, M5={identita_m5}"
        )

    posizione_m4 = calcola_posizione(r4)
    posizione_m5 = calcola_posizione(r5)

    posizioni_m4.append(posizione_m4)
    posizioni_m5.append(posizione_m5)

    identita = identita_m4 or identita_m5

    righe_dettaglio.append({
        "document_id": document_id,
        "identita_vera": identita,
        "posizione_m4": posizione_m4,
        "posizione_m5": posizione_m5,
        "identita_trovata_m4": int(posizione_m4 <= 10),
        "identita_trovata_m5": int(posizione_m5 <= 10),
        "variazione_posizione_m5_meno_m4": posizione_m5 - posizione_m4,
        "top1_m4": int(posizione_m4 <= 1),
        "top3_m4": int(posizione_m4 <= 3),
        "top5_m4": int(posizione_m4 <= 5),
        "top10_m4": int(posizione_m4 <= 10),
        "mrr_contributo_m4": reciprocal_rank(posizione_m4),
        "top1_m5": int(posizione_m5 <= 1),
        "top3_m5": int(posizione_m5 <= 3),
        "top5_m5": int(posizione_m5 <= 5),
        "top10_m5": int(posizione_m5 <= 10),
        "mrr_contributo_m5": reciprocal_rank(posizione_m5),
    })


# ============================================================
# CALCOLO METRICHE
# ============================================================

metriche_m4 = calcola_metriche(posizioni_m4)
metriche_m5 = calcola_metriche(posizioni_m5)

righe_riepilogo = [
    "CONFRONTO M4 vs M5 - STESSI 11 DOCUMENTI",
    "=" * 55,
    f"Documenti comuni: {len(ids_comuni)}",
    "Posizione 11 = identità non recuperata nella Top 10.",
    "Per questi casi il contributo all'MRR è zero.",
    "",
    "METRICHE M4",
]

for k in (1, 3, 5, 10):
    valore = metriche_m4[f"Top{k}"]
    righe_riepilogo.append(
        f"Top{k}: {valore['successi']}/{len(ids_comuni)} "
        f"({valore['percentuale']:.1f}%)"
    )

righe_riepilogo.append(f"MRR: {metriche_m4['MRR']:.4f}")
righe_riepilogo.extend(["", "METRICHE M5"])

for k in (1, 3, 5, 10):
    valore = metriche_m5[f"Top{k}"]
    righe_riepilogo.append(
        f"Top{k}: {valore['successi']}/{len(ids_comuni)} "
        f"({valore['percentuale']:.1f}%)"
    )

righe_riepilogo.append(f"MRR: {metriche_m5['MRR']:.4f}")
righe_riepilogo.extend(["", "DIFFERENZA M5 - M4"])

for k in (1, 3, 5, 10):
    differenza = (
        metriche_m5[f"Top{k}"]["percentuale"]
        - metriche_m4[f"Top{k}"]["percentuale"]
    )
    righe_riepilogo.append(f"Top{k}: {differenza:+.1f} punti percentuali")

differenza_mrr = metriche_m5["MRR"] - metriche_m4["MRR"]
righe_riepilogo.append(f"MRR: {differenza_mrr:+.4f}")


# ============================================================
# SALVATAGGIO
# ============================================================

with OUTPUT_DETTAGLIO.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    campi = list(righe_dettaglio[0].keys())
    writer = csv.DictWriter(f, fieldnames=campi)
    writer.writeheader()
    writer.writerows(righe_dettaglio)

with OUTPUT_RIEPILOGO.open("w", encoding="utf-8") as f:
    f.write("\n".join(righe_riepilogo) + "\n")

print("\n".join(righe_riepilogo))
print(f"\nDettaglio salvato in: {OUTPUT_DETTAGLIO}")
print(f"Riepilogo salvato in: {OUTPUT_RIEPILOGO}")


