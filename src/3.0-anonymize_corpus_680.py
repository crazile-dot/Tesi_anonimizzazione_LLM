"""
3.0-anonymize_corpus_680.py

Scopo:
Generare le quattro condizioni sperimentali M0, M1, M2 e M3
per tutti i documenti presenti in email_manifest.csv.

Input:
- data/processed/email_manifest.csv
- data/raw/enronRandom-XML/<relative_path>

Output:
- data/processed/dataset_680_baseline.jsonl

Strategie:
M0: testo originale, con i tag XML rimossi.
M1: redaction di persone, email e Message-ID.
M2: pseudonimizzazione deterministica per categoria.
M3: pseudonimizzazione mediante HMAC-SHA256.

Le occorrenze dei nomi annotati vengono sostituite anche fuori
dai tag XML, comprese le intestazioni delle email.

Le annotazioni contenenti più nomi vengono mantenute indivise.
Non vengono eseguiti rilevamento automatico di nuovi nomi
né anonimizzazione mediante LLM.
"""

from pathlib import Path
import csv
import json
import re
import hashlib
import hmac


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    PROJECT_PATH
    / "data"
    / "processed"
    / "email_manifest.csv"
)

RAW_DIR = (
    PROJECT_PATH
    / "data"
    / "raw"
    / "enronRandom-XML"
)

OUTPUT_FILE = (
    PROJECT_PATH
    / "data"
    / "processed"
    / "dataset_680_baseline.jsonl"
)

# Chiave sperimentale M3.
# Non viene salvata nel dataset di output.
HMAC_SECRET_KEY = b"tesi-anonimizzazione-secret-key"

TAG_PATTERN = re.compile(
    r"<true_name>(.*?)</true_name>",
    re.IGNORECASE | re.DOTALL
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE
)

MESSAGE_ID_PATTERN = re.compile(
    r"(?im)^(Message-ID\s*:\s*)(.*)$"
)


# ============================================================
# NORMALIZZAZIONE
# ============================================================

def normalizza_valore(valore):
    """Normalizza spazi e maiuscole/minuscole."""
    return " ".join(valore.split()).casefold()


# ============================================================
# LETTURA DEL DOCUMENTO E DELLE ANNOTAZIONI
# ============================================================

def leggi_documento(relative_path):
    """
    Legge il documento originale, rimuove i tag <true_name>
    e restituisce il testo pulito con gli offset delle PERSON.

    Il contenuto delle annotazioni viene mantenuto indiviso,
    anche quando contiene più nomi.
    """

    file_path = RAW_DIR / Path(relative_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Documento non trovato: {file_path}"
        )

    raw_text = file_path.read_text(
        encoding="latin-1",
        errors="replace"
    )

    parti = []
    entita = []
    posizione = 0
    lunghezza_testo = 0

    for match in TAG_PATTERN.finditer(raw_text):
        # Testo che precede il tag
        prefisso = raw_text[posizione:match.start()]
        parti.append(prefisso)
        lunghezza_testo += len(prefisso)

        # Contenuto originale del tag
        contenuto = match.group(1)

        # Gli spazi esterni restano nel testo, ma non
        # fanno parte dell'intervallo annotato.
        contenuto_pulito = contenuto.strip()

        spazi_iniziali = (
            len(contenuto) - len(contenuto.lstrip())
        )

        inizio = lunghezza_testo + spazi_iniziali
        fine = inizio + len(contenuto_pulito)

        # Il contenuto viene mantenuto nel testo senza i tag.
        parti.append(contenuto)
        lunghezza_testo += len(contenuto)

        if contenuto_pulito:
            entita.append({
                "start": inizio,
                "end": fine,
                "text": contenuto_pulito
            })

        posizione = match.end()

    # Testo successivo all'ultimo tag
    parti.append(raw_text[posizione:])

    testo = "".join(parti)

    return testo, entita


# ============================================================
# GENERAZIONE DEI CANDIDATI DI SOSTITUZIONE
# ============================================================

def crea_candidati(testo, entita):
    """
    Individua:
    - entità PERSON annotate;
    - occorrenze degli stessi nomi fuori dai tag;
    - indirizzi email;
    - intestazioni Message-ID.

    Le sostituzioni sovrapposte vengono risolte dando priorità
    a Message-ID, poi email e infine PERSON.
    """

    candidati = []

    # 1. PERSON annotate e relativi nomi normalizzati
    nomi = {}

    for entity in entita:
        originale = entity["text"]
        chiave = normalizza_valore(originale)

        if chiave:
            nomi.setdefault(chiave, originale)

        candidati.append({
            "start": entity["start"],
            "end": entity["end"],
            "category": "PERSON",
            "key": chiave,
            "priority": 1
        })

    # 2. Occorrenze dei nomi annotati nel resto del testo
    # I nomi più lunghi vengono cercati per primi.
    for chiave, originale in sorted(
        nomi.items(),
        key=lambda item: len(item[1]),
        reverse=True
    ):
        pattern = re.compile(
            r"(?<!\w)"
            + re.escape(originale.strip())
            + r"(?!\w)",
            re.IGNORECASE
        )

        for match in pattern.finditer(testo):
            candidati.append({
                "start": match.start(),
                "end": match.end(),
                "category": "PERSON",
                "key": chiave,
                "priority": 1
            })

    # 3. Indirizzi email
    for match in EMAIL_PATTERN.finditer(testo):
        valore = match.group(0)

        candidati.append({
            "start": match.start(),
            "end": match.end(),
            "category": "EMAIL",
            "key": normalizza_valore(valore),
            "priority": 2
        })

    # 4. Message-ID: si sostituisce il valore, non il nome
    # dell'intestazione "Message-ID:".
    for match in MESSAGE_ID_PATTERN.finditer(testo):
        valore = match.group(2)

        if valore.strip():
            candidati.append({
                "start": match.start(2),
                "end": match.end(2),
                "category": "MESSAGE_ID",
                "key": normalizza_valore(valore),
                "priority": 3
            })

    return candidati


# ============================================================
# RISOLUZIONE DELLE SOVRAPPOSIZIONI
# ============================================================

def seleziona_candidati(candidati):
    """
    Seleziona intervalli non sovrapposti.

    Priorità:
    1. MESSAGE_ID
    2. EMAIL
    3. PERSON

    A parità di priorità, vengono considerati prima gli
    intervalli più lunghi.
    """

    ordinati = sorted(
        candidati,
        key=lambda c: (
            -c["priority"],
            -(c["end"] - c["start"]),
            c["start"]
        )
    )

    selezionati = []

    for candidato in ordinati:
        sovrapposto = any(
            candidato["start"] < scelto["end"]
            and candidato["end"] > scelto["start"]
            for scelto in selezionati
        )

        if not sovrapposto:
            selezionati.append(candidato)

    return sorted(
        selezionati,
        key=lambda c: c["start"]
    )


# ============================================================
# APPLICAZIONE DELLE SOSTITUZIONI
# ============================================================

def applica_sostituzioni(testo, sostituzioni):
    """Applica le sostituzioni da destra verso sinistra."""

    risultato = testo

    for sostituzione in sorted(
        sostituzioni,
        key=lambda s: s["start"],
        reverse=True
    ):
        risultato = (
            risultato[:sostituzione["start"]]
            + sostituzione["replacement"]
            + risultato[sostituzione["end"]:]
        )

    return risultato


# ============================================================
# M1 - REDACTION
# ============================================================

def crea_m1(testo, candidati):
    """Sostituisce PERSON, email e Message-ID con placeholder."""

    sostituzioni = []

    placeholder = {
        "PERSON": "[REDACTED]",
        "EMAIL": "[EMAIL]",
        "MESSAGE_ID": "[MESSAGE_ID]"
    }

    for candidato in seleziona_candidati(candidati):
        sostituzioni.append({
            "start": candidato["start"],
            "end": candidato["end"],
            "replacement": placeholder[candidato["category"]]
        })

    return applica_sostituzioni(testo, sostituzioni)


# ============================================================
# M2 - PSEUDONIMIZZAZIONE DETERMINISTICA
# ============================================================

def crea_m2(testo, candidati):
    """
    Assegna identificativi coerenti all'interno del documento.

    La numerazione è separata per categoria:
    PERSON_1, EMAIL_1, MESSAGE_ID_1, ecc.
    """

    contatori = {
        "PERSON": 0,
        "EMAIL": 0,
        "MESSAGE_ID": 0
    }

    mappe = {
        "PERSON": {},
        "EMAIL": {},
        "MESSAGE_ID": {}
    }

    mappa_output = []
    sostituzioni = []

    prefissi = {
        "PERSON": "PERSON",
        "EMAIL": "EMAIL",
        "MESSAGE_ID": "MESSAGE_ID"
    }

    for candidato in seleziona_candidati(candidati):
        categoria = candidato["category"]
        chiave = candidato["key"]

        if chiave not in mappe[categoria]:
            contatori[categoria] += 1

            if categoria == "PERSON":
                replacement = (
                    f"PERSON_{contatori[categoria]}"
                )
            else:
                replacement = (
                    f"[{prefissi[categoria]}_"
                    f"{contatori[categoria]}]"
                )

            mappe[categoria][chiave] = replacement

            mappa_output.append({
                "category": categoria,
                "original_normalized": chiave,
                "replacement": replacement
            })

        replacement = mappe[categoria][chiave]

        sostituzioni.append({
            "start": candidato["start"],
            "end": candidato["end"],
            "replacement": replacement
        })

    risultato = applica_sostituzioni(
        testo,
        sostituzioni
    )

    return risultato, mappa_output


# ============================================================
# M3 - PSEUDONIMIZZAZIONE MEDIANTE HMAC-SHA256
# ============================================================

def crea_m3(testo, candidati):
    """
    Calcola l'HMAC a partire da categoria e valore normalizzato.
    La stessa entità produce lo stesso identificativo
    all'interno del documento e tra documenti.
    """

    mappa = {}
    mappa_output = []
    sostituzioni = []

    for candidato in seleziona_candidati(candidati):
        categoria = candidato["category"]
        chiave = candidato["key"]

        chiave_hmac = f"{categoria}:{chiave}"

        if chiave_hmac not in mappa:
            digest = hmac.new(
                HMAC_SECRET_KEY,
                chiave_hmac.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()[:12]

            if categoria == "PERSON":
                replacement = f"PERSON_{digest}"
            else:
                replacement = f"[{categoria}_{digest}]"

            mappa[chiave_hmac] = replacement

            mappa_output.append({
                "category": categoria,
                "original_normalized": chiave,
                "replacement": replacement
            })

        sostituzioni.append({
            "start": candidato["start"],
            "end": candidato["end"],
            "replacement": mappa[chiave_hmac]
        })

    risultato = applica_sostituzioni(
        testo,
        sostituzioni
    )

    return risultato, mappa_output


# ============================================================
# ELABORAZIONE DI UN DOCUMENTO
# ============================================================

def anonimizza_documento(document_id, split, relative_path):
    testo, entita = leggi_documento(relative_path)

    candidati = crea_candidati(testo, entita)

    testo_m1 = crea_m1(testo, candidati)

    testo_m2, mappa_m2 = crea_m2(
        testo,
        candidati
    )

    testo_m3, mappa_m3 = crea_m3(
        testo,
        candidati
    )

    conteggi = {
        "PERSON_annotate": len(entita),
        "PERSON_sostituite": sum(
            1 for c in seleziona_candidati(candidati)
            if c["category"] == "PERSON"
        ),
        "EMAIL_sostituite": sum(
            1 for c in seleziona_candidati(candidati)
            if c["category"] == "EMAIL"
        ),
        "MESSAGE_ID_sostituiti": sum(
            1 for c in seleziona_candidati(candidati)
            if c["category"] == "MESSAGE_ID"
        )
    }

    return {
        "document_id": document_id,
        "split": split,
        "relative_path": relative_path,

        "testo_originale": testo,
        "M0_originale": testo,

        "M1_redacted": testo_m1,

        "M2_persona": testo_m2,
        "M2_anonymization_map": mappa_m2,

        "M3_hmac": testo_m3,
        "M3_anonymization_map": mappa_m3,

        "conteggi": conteggi,
        "numero_entita_persona_annotate": len(entita)
    }


# ============================================================
# MAIN
# ============================================================

def main():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest non trovato: {MANIFEST_PATH}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    records_processed = 0
    total_person = 0
    total_email = 0
    total_message_id = 0

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as manifest_file:

        reader = csv.DictReader(manifest_file)

        colonne_richieste = {
            "document_id",
            "relative_path",
            "split"
        }

        colonne_disponibili = set(reader.fieldnames or [])
        mancanti = colonne_richieste - colonne_disponibili

        if mancanti:
            raise ValueError(
                "Colonne mancanti in email_manifest.csv: "
                + ", ".join(sorted(mancanti))
            )

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as outfile:

            for row in reader:
                document_id = row["document_id"]
                relative_path = row["relative_path"]
                split = row["split"]

                result = anonimizza_documento(
                    document_id,
                    split,
                    relative_path
                )

                outfile.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    ) + "\n"
                )

                records_processed += 1
                conteggi = result["conteggi"]

                total_person += conteggi["PERSON_annotate"]
                total_email += conteggi["EMAIL_sostituite"]
                total_message_id += conteggi["MESSAGE_ID_sostituiti"]

                if records_processed % 50 == 0:
                    print(
                        f"Documenti elaborati: "
                        f"{records_processed}"
                    )

    print("\n--- BASELINE CREATA ---")
    print(f"Documenti elaborati: {records_processed}")
    print(f"Entità PERSON annotate: {total_person}")
    print(f"Occorrenze email sostituite: {total_email}")
    print(f"Message-ID sostituiti: {total_message_id}")
    print(f"\nFile salvato in:\n{OUTPUT_FILE}")


# ============================================================
# AVVIO
# ============================================================

if __name__ == "__main__":
    main()