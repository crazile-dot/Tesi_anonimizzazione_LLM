
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "direct_leakage_per_entity_680_v2.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

METODI = [
    "M4 - LLM substitution",
    "M5 - LLM context",
]

COLONNE_NUMERICHE = [
    "menzioni_originali",
    "menzioni_residue",
    "direct_leakage_rate_percent",
    "documenti_con_entita",
    "documenti_con_leakage",
    "document_leakage_rate_percent",
]


# ============================================================
# LETTURA E PREPARAZIONE
# ============================================================

def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"CSV non trovato: {INPUT_CSV}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    colonne_richieste = {
        "metodo",
        "entita_annotata",
        *COLONNE_NUMERICHE,
    }

    mancanti = colonne_richieste - set(df.columns)

    if mancanti:
        raise ValueError(
            f"Colonne mancanti nel CSV: {sorted(mancanti)}"
        )

    df = df[df["metodo"].isin(METODI)].copy()

    if df.empty:
        raise ValueError(
            "Non sono state trovate righe per M4 e M5. "
            "Controlla i valori della colonna 'metodo'."
        )

    for col in COLONNE_NUMERICHE:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if (df["menzioni_originali"] < 0).any():
        raise ValueError("Trovate menzioni originali negative.")

    if (df["menzioni_residue"] < 0).any():
        raise ValueError("Trovate menzioni residue negative.")

    if (
        df["menzioni_residue"] > df["menzioni_originali"]
    ).any():
        raise ValueError(
            "Alcune entità hanno più residui che menzioni originali."
        )

    # Elimina eventuali righe duplicate metodo-entità.
    duplicati = df.duplicated(
        subset=["metodo", "entita_annotata"],
        keep=False,
    )

    if duplicati.any():
        esempi = df.loc[
            duplicati, ["metodo", "entita_annotata"]
        ].head(10)

        raise ValueError(
            "Ci sono righe duplicate per metodo-entità. "
            "Verifica il CSV prima di aggregare.\n"
            + esempi.to_string(index=False)
        )

    # Ricalcola il tasso a partire dai conteggi:
    # evita di dipendere da percentuali arrotondate nel CSV.
    df["tasso_ricalcolato"] = np.where(
        df["menzioni_originali"] > 0,
        100 * df["menzioni_residue"] / df["menzioni_originali"],
        np.nan,
    )

    df["leakage_presente"] = df["menzioni_residue"] > 0
    df["leakage_almeno_50"] = df["tasso_ricalcolato"] >= 50


    # ========================================================
    # 1. STATISTICHE AGGREGATE PER METODO
    # ========================================================

    righe = []

    for metodo in METODI:
        sotto = df[df["metodo"] == metodo].copy()

        if sotto.empty:
            continue

        menzioni_originali = sotto["menzioni_originali"].sum()
        menzioni_residue = sotto["menzioni_residue"].sum()

        entita = len(sotto)
        entita_con_leakage = int(sotto["leakage_presente"].sum())
        entita_senza_leakage = entita - entita_con_leakage

        documenti_entita = sotto["documenti_con_entita"].sum()
        documenti_leakage = sotto["documenti_con_leakage"].sum()

        tassi = sotto["tasso_ricalcolato"].dropna()

        righe.append({
            "metodo": metodo,
            "entita_distinte": entita,
            "menzioni_originali": int(menzioni_originali),
            "menzioni_residue": int(menzioni_residue),

            "direct_leakage_ponderato_percent": (
                100 * menzioni_residue / menzioni_originali
                if menzioni_originali else np.nan
            ),

            "tasso_medio_entita_percent": tassi.mean(),
            "tasso_mediano_entita_percent": tassi.median(),
            "deviazione_standard_tassi_percent": tassi.std(ddof=1),
            "primo_quartile_percent": tassi.quantile(0.25),
            "terzo_quartile_percent": tassi.quantile(0.75),

            "entita_senza_residui": entita_senza_leakage,
            "percentuale_entita_senza_residui": (
                100 * entita_senza_leakage / entita
                if entita else np.nan
            ),

            "entita_con_residui": entita_con_leakage,
            "percentuale_entita_con_residui": (
                100 * entita_con_leakage / entita
                if entita else np.nan
            ),

            "entita_leakage_almeno_50_percento": int(
                sotto["leakage_almeno_50"].sum()
            ),
            "percentuale_entita_leakage_almeno_50": (
                100 * sotto["leakage_almeno_50"].sum() / entita
                if entita else np.nan
            ),

            "somma_documenti_con_entita": int(documenti_entita),
            "somma_documenti_con_leakage": int(documenti_leakage),

            "quota_documenti_con_leakage_percent": (
                100 * documenti_leakage / documenti_entita
                if documenti_entita else np.nan
            ),
        })

    statistiche = pd.DataFrame(righe)


    # ========================================================
    # 2. CONFRONTO M4-M5 SULLE STESSE ENTITÀ
    # ========================================================

    m4 = (
        df[df["metodo"] == METODI[0]]
        .set_index("entita_annotata")
    )

    m5 = (
        df[df["metodo"] == METODI[1]]
        .set_index("entita_annotata")
    )

    # Confronto principale sulle entità presenti in entrambi i metodi.
    comuni = m4.index.intersection(m5.index)

    confronto = pd.DataFrame(index=comuni)

    confronto["menzioni_originali_m4"] = (
        m4.loc[comuni, "menzioni_originali"]
    )
    confronto["menzioni_originali_m5"] = (
        m5.loc[comuni, "menzioni_originali"]
    )
    confronto["residue_m4"] = m4.loc[comuni, "menzioni_residue"]
    confronto["residue_m5"] = m5.loc[comuni, "menzioni_residue"]

    confronto["tasso_m4"] = m4.loc[comuni, "tasso_ricalcolato"]
    confronto["tasso_m5"] = m5.loc[comuni, "tasso_ricalcolato"]

    # Le frequenze originali dovrebbero coincidere tra metodi.
    confronto["frequenza_originale_coerente"] = (
        confronto["menzioni_originali_m4"]
        == confronto["menzioni_originali_m5"]
    )

    # Differenza in punti percentuali: negativa = miglioramento M5.
    confronto["differenza_m5_meno_m4_pp"] = (
        confronto["tasso_m5"] - confronto["tasso_m4"]
    )

    confronto["esito"] = np.select(
        [
            confronto["differenza_m5_meno_m4_pp"] < -1e-9,
            confronto["differenza_m5_meno_m4_pp"] > 1e-9,
        ],
        [
            "Migliora con M5",
            "Peggiora con M5",
        ],
        default="Invariata",
    )

    confronto = confronto.reset_index(
        names="entita_annotata"
    )

    # Sintesi sulle sole entità comuni.
    conteggi_esito = (
        confronto["esito"]
        .value_counts()
        .to_dict()
    )

    numero_comuni = len(confronto)

    sintesi_confronto = pd.DataFrame([{
        "entita_m4": len(m4),
        "entita_m5": len(m5),
        "entita_comuni": numero_comuni,
        "entita_solo_m4": len(m4.index.difference(m5.index)),
        "entita_solo_m5": len(m5.index.difference(m4.index)),

        "entita_con_frequenza_originale_coerente": int(
            confronto["frequenza_originale_coerente"].sum()
        ),

        "migliora_con_m5": conteggi_esito.get(
            "Migliora con M5", 0
        ),
        "peggiora_con_m5": conteggi_esito.get(
            "Peggiora con M5", 0
        ),
        "invariata": conteggi_esito.get(
            "Invariata", 0
        ),

        "percentuale_migliorate_su_comuni": (
            100 * conteggi_esito.get("Migliora con M5", 0)
            / numero_comuni
            if numero_comuni else np.nan
        ),

        "mediana_differenza_m5_meno_m4_pp": (
            confronto["differenza_m5_meno_m4_pp"].median()
            if numero_comuni else np.nan
        ),

        "media_differenza_m5_meno_m4_pp": (
            confronto["differenza_m5_meno_m4_pp"].mean()
            if numero_comuni else np.nan
        ),
    }])


    # ========================================================
    # 3. CONTROLLI DI COERENZA
    # ========================================================

    # Per ogni metodo, il tasso ponderato dovrebbe corrispondere
    # al rapporto tra somma dei residui e somma delle menzioni.
    print("\n--- CONTROLLI ---")
    print(f"Righe analizzate M4/M5: {len(df)}")
    print(f"Entità presenti in entrambi i metodi: {numero_comuni}")

    incoerenti = confronto[
        ~confronto["frequenza_originale_coerente"]
    ]

    print(
        "Entità comuni con frequenza originale diversa tra M4 e M5: "
        f"{len(incoerenti)}"
    )

    if len(incoerenti):
        print(
            "ATTENZIONE: controllare le entità con frequenze diverse. "
            "Il confronto dei tassi potrebbe non essere omogeneo."
        )


    # ========================================================
    # 4. ESPORTAZIONE DEI RISULTATI
    # ========================================================

    file_statistiche = OUTPUT_DIR / "statistiche_direct_leakage_m4_m5.csv"
    file_confronto = OUTPUT_DIR / "confronto_entita_m4_m5.csv"
    file_sintesi = OUTPUT_DIR / "sintesi_confronto_m4_m5.csv"

    statistiche.to_csv(
        file_statistiche,
        index=False,
        encoding="utf-8-sig",
    )

    confronto.to_csv(
        file_confronto,
        index=False,
        encoding="utf-8-sig",
    )

    sintesi_confronto.to_csv(
        file_sintesi,
        index=False,
        encoding="utf-8-sig",
    )

    # Stampa riepilogo leggibile nel terminale.
    colonne_stampa = [
        "metodo",
        "entita_distinte",
        "menzioni_originali",
        "menzioni_residue",
        "direct_leakage_ponderato_percent",
        "tasso_medio_entita_percent",
        "tasso_mediano_entita_percent",
        "percentuale_entita_senza_residui",
        "percentuale_entita_con_residui",
    ]

    print("\n--- STATISTICHE AGGREGATE M4/M5 ---")
    print(
        statistiche[colonne_stampa]
        .round(2)
        .to_string(index=False)
    )

    print("\n--- SINTESI CONFRONTO SULLE ENTITÀ COMUNI ---")
    print(sintesi_confronto.round(2).to_string(index=False))

    print("\nFile generati:")
    print(file_statistiche)
    print(file_confronto)
    print(file_sintesi)


if __name__ == "__main__":
    main()
