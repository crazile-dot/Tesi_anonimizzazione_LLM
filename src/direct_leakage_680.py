
"""
Valuta il direct leakage su M1-M5.
Conta le occorrenze annotate residue e i documenti
che contengono almeno un identificatore annotato residuo.
"""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import re


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MANIFEST_FILE = PROJECT_DIR / "data" / "processed" / "email_manifest.csv"
RAW_DIR = PROJECT_DIR / "data" / "raw" / "enronRandom-XML"
BASELINE_FILE = PROJECT_DIR / "data" / "processed" / "dataset_680_baseline.jsonl"
M4_FILE = PROJECT_DIR / "data" / "processed" / "openai_results_680_m4_v2.jsonl"
M5_FILE = PROJECT_DIR / "data" / "processed" / "openai_results_680_m5_v2.jsonl"
OUTPUT_CSV = PROJECT_DIR / "data" / "processed" / "direct_leakage_680_v2.csv"

TAG_PATTERN = re.compile(
    r"<true_name>(.*?)</true_name>",
    re.IGNORECASE | re.DOTALL,
)


# ============================================================
# LETTURA FILE
# ============================================================

def leggi_jsonl(percorso):
    records = {}

    with percorso.open("r", encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue

            record = json.loads(linea)
            document_id = record.get("document_id")

            if document_id:
                records[document_id] = record

    return records


def leggi_risultati_llm(percorso):
    """Carica l'ultimo risultato valido per ogni documento."""

    if not percorso.is_file():
        raise FileNotFoundError(
            f"File risultati non trovato: {percorso}"
        )

    records = {}

    with percorso.open("r", encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue

            record = json.loads(linea)
            document_id = record.get("document_id")

            if (
                document_id
                and record.get("status") == "success"
                and isinstance(record.get("text_anonymized"), str)
                and record["text_anonymized"].strip()
            ):
                records[document_id] = record

    return records


def leggi_menzioni(relative_path):
    """Estrae le annotazioni XML senza dividerne il contenuto."""

    percorso = RAW_DIR / Path(relative_path)

    if not percorso.is_file():
        raise FileNotFoundError(percorso)

    testo = percorso.read_text(
        encoding="latin-1",
        errors="replace",
    )

    return [
        match.group(1).strip()
        for match in TAG_PATTERN.finditer(testo)
        if match.group(1).strip()
    ]


# ============================================================
# CONTEGGIO DELLE OCCORRENZE RESIDUE
# ============================================================

def conta_occorrenze(nome, testo):
    """Conta le corrispondenze complete, ignorando maiuscole/minuscole."""

    pattern = re.compile(
        r"(?<!\w)" + re.escape(nome) + r"(?!\w)",
        re.IGNORECASE,
    )

    return sum(1 for _ in pattern.finditer(testo))


def valuta_metodo(documenti, testi_anonimizzati):
    documenti_valutati = 0
    documenti_con_leakage = 0
    occorrenze_controllate = 0
    occorrenze_residue = 0
    documenti_senza_risultato = []

    for doc in documenti:
        document_id = doc["document_id"]

        if document_id not in testi_anonimizzati:
            documenti_senza_risultato.append(document_id)
            continue

        testo = testi_anonimizzati[document_id]
        menzioni = doc["menzioni"]

        documenti_valutati += 1
        occorrenze_controllate += len(menzioni)

        # Frequenza delle annotazioni identiche nel documento originale.
        frequenze_originali = Counter(menzioni)
        residue_documento = 0

        for nome, frequenza_originale in frequenze_originali.items():
            frequenza_residua = conta_occorrenze(nome, testo)

            # Non attribuiamo al nome più occorrenze residue
            # di quante ne risultino annotate nell'originale.
            residue_documento += min(
                frequenza_originale,
                frequenza_residua,
            )

        occorrenze_residue += residue_documento

        if residue_documento > 0:
            documenti_con_leakage += 1

    leakage_rate = (
        100 * occorrenze_residue / occorrenze_controllate
        if occorrenze_controllate else None
    )

    document_rate = (
        100 * documenti_con_leakage / documenti_valutati
        if documenti_valutati else None
    )

    return {
        "documenti_valutati": documenti_valutati,
        "documenti_senza_risultato": len(documenti_senza_risultato),
        "occorrenze_controllate": occorrenze_controllate,
        "occorrenze_residue": occorrenze_residue,
        "direct_leakage_rate_percent": (
            round(leakage_rate, 2) if leakage_rate is not None else None
        ),
        "documenti_con_leakage": documenti_con_leakage,
        "document_leakage_rate_percent": (
            round(document_rate, 2) if document_rate is not None else None
        ),
        "documenti_esclusi": documenti_senza_risultato,
    }


OUTPUT_ENTITY_CSV = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "direct_leakage_per_entity_680_v2.csv"
)


def normalizza_entita(nome):
    """Uniforma maiuscole/minuscole e spazi per aggregare le varianti."""
    return re.sub(r"\s+", " ", nome.strip()).casefold()



def esporta_direct_leakage_per_entita(
    documenti,
    testi_anonimizzati_per_metodo,
    output_csv=OUTPUT_ENTITY_CSV,
):
    """
    Produce una riga per ogni metodo e forma nominale annotata.

    Le varianti che differiscono soltanto per maiuscole/minuscole
    o spazi vengono aggregate. Le annotazioni XML non vengono divise.
    """

    aggregati = defaultdict(lambda: {
        "nome": "",
        "menzioni_originali": 0,
        "menzioni_residue": 0,
        "documenti_con_entita": 0,
        "documenti_con_leakage": 0,
    })

    for metodo, testi_anonimizzati in testi_anonimizzati_per_metodo.items():
        for doc in documenti:
            document_id = doc["document_id"]

            # Esclude i documenti senza risultato per il metodo corrente.
            if document_id not in testi_anonimizzati:
                continue

            testo_anonimizzato = testi_anonimizzati[document_id]
            menzioni = doc["menzioni"]

            # Raggruppa le varianti equivalenti all'interno del documento.
            menzioni_per_entita = defaultdict(Counter)

            for nome in menzioni:
                nome_normalizzato = normalizza_entita(nome)

                if nome_normalizzato:
                    menzioni_per_entita[nome_normalizzato][nome] += 1

            for chiave_entita, varianti in menzioni_per_entita.items():
                # La chiave comprende metodo ed entità:
                # i risultati dei diversi metodi restano separati.
                chiave_aggregato = (metodo, chiave_entita)
                record = aggregati[chiave_aggregato]

                nome_rappresentativo = next(iter(varianti))
                occorrenze_originali = sum(varianti.values())

                if not record["nome"]:
                    record["nome"] = nome_rappresentativo

                # Costruisce un'unica regex con tutte le varianti,
                # ordinate dalla più lunga alla più corta.
                # Gli spazi multipli vengono trattati come equivalenti.
                varianti_ordinate = sorted(
                    varianti.keys(),
                    key=len,
                    reverse=True,
                )

                alternative = []

                for variante in varianti_ordinate:
                    parti = re.split(r"\s+", variante.strip())
                    alternativa = r"\s+".join(
                        re.escape(parte) for parte in parti
                    )
                    alternative.append(alternativa)

                pattern = re.compile(
                    r"(?<!\w)(?:"
                    + "|".join(alternative)
                    + r")(?!\w)",
                    re.IGNORECASE,
                )

                # Conta le corrispondenze senza contare due volte
                # la stessa posizione per varianti sovrapposte.
                frequenza_residua = sum(
                    1 for _ in pattern.finditer(testo_anonimizzato)
                )

                # Non attribuisce più residui di quanti siano annotati.
                residue = min(
                    occorrenze_originali,
                    frequenza_residua,
                )

                record["menzioni_originali"] += occorrenze_originali
                record["menzioni_residue"] += residue
                record["documenti_con_entita"] += 1

                if residue > 0:
                    record["documenti_con_leakage"] += 1

    colonne = [
        "metodo",
        "entita_annotata",
        "menzioni_originali",
        "menzioni_residue",
        "direct_leakage_rate_percent",
        "documenti_con_entita",
        "documenti_con_leakage",
        "document_leakage_rate_percent",
    ]

    righe = []

    for (metodo, _), record in aggregati.items():
        originali = record["menzioni_originali"]
        residue = record["menzioni_residue"]
        documenti_entita = record["documenti_con_entita"]
        documenti_leakage = record["documenti_con_leakage"]

        righe.append({
            "metodo": metodo,
            "entita_annotata": record["nome"],
            "menzioni_originali": originali,
            "menzioni_residue": residue,
            "direct_leakage_rate_percent": (
                round(100 * residue / originali, 4)
                if originali else 0.0
            ),
            "documenti_con_entita": documenti_entita,
            "documenti_con_leakage": documenti_leakage,
            "document_leakage_rate_percent": (
                round(100 * documenti_leakage / documenti_entita, 4)
                if documenti_entita else 0.0
            ),
        })

    # Ordina per metodo e leakage decrescente.
    righe.sort(
        key=lambda r: (
            r["metodo"],
            -r["menzioni_residue"],
            -r["direct_leakage_rate_percent"],
            r["entita_annotata"].casefold(),
        )
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=colonne)
        writer.writeheader()
        writer.writerows(righe)

    print(f"Dettaglio per entità salvato in: {output_csv}")
    print(f"Righe generate: {len(righe)}")



# ============================================================
# MAIN
# ============================================================

def main():
    for percorso in (MANIFEST_FILE, BASELINE_FILE, M4_FILE, M5_FILE):
        if not percorso.is_file():
            raise FileNotFoundError(
                f"File necessario non trovato: {percorso}"
            )

    with MANIFEST_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        manifest = list(csv.DictReader(f))

    baseline = leggi_jsonl(BASELINE_FILE)
    m4 = leggi_risultati_llm(M4_FILE)
    m5 = leggi_risultati_llm(M5_FILE)

    documenti = []

    for row in manifest:
        document_id = row["document_id"]
        relative_path = row["relative_path"]

        if document_id not in baseline:
            print(f"ATTENZIONE: {document_id} non è nel baseline.")
            continue

        try:
            menzioni = leggi_menzioni(relative_path)
        except Exception as exc:
            print(f"ATTENZIONE: impossibile leggere {document_id}: {exc}")
            continue

        documenti.append({
            "document_id": document_id,
            "relative_path": relative_path,
            "menzioni": menzioni,
        })

    metodi = {
        "M1 - Redaction": {
            k: v["M1_redacted"]
            for k, v in baseline.items()
            if isinstance(v.get("M1_redacted"), str)
        },
        "M2 - Pseudonimizzazione": {
            k: v["M2_persona"]
            for k, v in baseline.items()
            if isinstance(v.get("M2_persona"), str)
        },
        "M3 - HMAC": {
            k: v["M3_hmac"]
            for k, v in baseline.items()
            if isinstance(v.get("M3_hmac"), str)
        },
        "M4 - LLM substitution": {
            k: v["text_anonymized"] for k, v in m4.items()
        },
        "M5 - LLM context": {
            k: v["text_anonymized"] for k, v in m5.items()
        },
    }

    risultati = []

    print("\n--- DIRECT LEAKAGE: CORPUS COMPLETO ---")

    for metodo, testi in metodi.items():
        risultato = valuta_metodo(documenti, testi)
        risultato["metodo"] = metodo
        risultati.append(risultato)

        print(f"\n{metodo}")
        print(f"Documenti valutati: {risultato['documenti_valutati']}")
        print(
            f"Documenti senza risultato: "
            f"{risultato['documenti_senza_risultato']}"
        )
        print(
            f"Occorrenze controllate: "
            f"{risultato['occorrenze_controllate']}"
        )
        print(
            f"Occorrenze residue: {risultato['occorrenze_residue']}"
        )
        print(
            f"Direct leakage rate: "
            f"{risultato['direct_leakage_rate_percent']}%"
        )
        print(
            f"Documenti con almeno un identificatore residuo: "
            f"{risultato['documenti_con_leakage']}"
        )
        print(
            f"Document leakage rate: "
            f"{risultato['document_leakage_rate_percent']}%"
        )

        if risultato["documenti_esclusi"]:
            print(
                "Documenti esclusi: "
                + ", ".join(risultato["documenti_esclusi"])
            )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    colonne = [
        "metodo",
        "documenti_valutati",
        "documenti_senza_risultato",
        "occorrenze_controllate",
        "occorrenze_residue",
        "direct_leakage_rate_percent",
        "documenti_con_leakage",
        "document_leakage_rate_percent",
    ]

    with OUTPUT_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=colonne)
        writer.writeheader()
        writer.writerows(
            {colonna: risultato[colonna] for colonna in colonne}
            for risultato in risultati
        )

    print(f"\nRisultati salvati in:\n{OUTPUT_CSV}")

    esporta_direct_leakage_per_entita(
        documenti=documenti,
        testi_anonimizzati_per_metodo=metodi,
    )


if __name__ == "__main__":
    main()
