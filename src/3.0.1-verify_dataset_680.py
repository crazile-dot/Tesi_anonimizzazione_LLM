
"""
3.0.1-verify_dataset_680.py

Verifica il dataset baseline generato da 3.0-anonymize_corpus_680.py.
Non modifica né rigenera il dataset.
"""

import json
import re
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "dataset_680_baseline.jsonl"
)

DOC_ID = "mclaughlin-e_all_documents_242"

PATTERNS = {
    "PERSON numerici M2": re.compile(r"\bPERSON_\d+\b"),
    "PERSON HMAC M3": re.compile(r"\bPERSON_[0-9a-f]{12}\b"),
    "EMAIL redatte M1": re.compile(r"\[EMAIL\]"),
    "EMAIL pseudonime M2": re.compile(r"\[EMAIL_\d+\]"),
    "EMAIL HMAC M3": re.compile(r"\[EMAIL_[0-9a-f]{12}\]"),
    "Message-ID redatti M1": re.compile(r"\[MESSAGE_ID\]"),
    "Message-ID pseudonimi M2": re.compile(r"\[MESSAGE_ID_\d+\]"),
    "Message-ID HMAC M3": re.compile(r"\[MESSAGE_ID_[0-9a-f]{12}\]"),
}


def conta(pattern, testo):
    return len(pattern.findall(testo))


def main():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset non trovato: {DATASET_PATH}"
        )

    documenti = []

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                documenti.append(json.loads(linea))

    print("--- VERIFICA DATASET ---")
    print("Documenti nel dataset:", len(documenti))

    if len(documenti) != 680:
        print("ATTENZIONE: il numero di documenti è diverso da 680.")

    doc = next(
        (d for d in documenti if d["document_id"] == DOC_ID),
        None
    )

    if doc is None:
        print(f"\nDocumento {DOC_ID} non trovato.")
        return

    print("\n--- DOCUMENTO DI CONTROLLO ---")
    print("ID:", doc["document_id"])
    print("Split:", doc["split"])
    print("PERSON annotate:", doc["numero_entita_persona_annotate"])
    print("Conteggi registrati:", doc["conteggi"])

    campi = {
        "M0": "M0_originale",
        "M1": "M1_redacted",
        "M2": "M2_persona",
        "M3": "M3_hmac",
    }

    print("\n--- LUNGHEZZA DEI TESTI ---")
    for strategia, campo in campi.items():
        print(f"{strategia}: {len(doc[campo])} caratteri")

    print("\n--- TOKEN NEL DOCUMENTO ---")
    for nome, pattern in PATTERNS.items():
        if "M1" in nome:
            campo = "M1_redacted"
        elif "M2" in nome:
            campo = "M2_persona"
        else:
            campo = "M3_hmac"

        print(f"{nome}: {conta(pattern, doc[campo])}")

    print("\n--- MAPPE DI PSEUDONIMIZZAZIONE ---")
    for campo in ("M2_anonymization_map", "M3_anonymization_map"):
        mappa = doc.get(campo, [])
        categorie = Counter(
            elemento["category"] for elemento in mappa
        )
        print(f"{campo}: {len(mappa)} identificativi distinti")
        print("  Per categoria:", dict(categorie))

    print("\n--- CONTROLLO PLACEHOLDER E CONTENUTO ---")
    print("M0 uguale a testo_originale:",
          doc["M0_originale"] == doc["testo_originale"])

    for strategia, campo in (
        ("M1", "M1_redacted"),
        ("M2", "M2_persona"),
        ("M3", "M3_hmac"),
    ):
        testo = doc[campo]
        print(
            f"{strategia}: contiene tag <true_name>:",
            bool(re.search(r"</?true_name>", testo, re.I))
        )
        print(
            f"{strategia}: contiene indirizzi email riconoscibili:",
            bool(re.search(
                r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                testo,
                re.I
            ))
        )
        print(
            f"{strategia}: contiene intestazioni Message-ID:",
            bool(re.search(r"(?im)^Message-ID\s*:", testo))
        )

    print("\nVerifica terminata. Il dataset non è stato modificato.")


if __name__ == "__main__":
    main()
