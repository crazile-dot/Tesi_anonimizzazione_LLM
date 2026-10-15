
from pathlib import Path
import json
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_DIR / "data" / "processed" / "corpus_test.jsonl"

tag_apertura = re.compile(r"<true_name>", re.IGNORECASE)
tag_chiusura = re.compile(r"</true_name>", re.IGNORECASE)

totale_aperture = 0
totale_chiusure = 0
documenti_sbilanciati = []
annotazioni_vuote = []
annotazioni_con_tag_interni = []

with CORPUS.open("r", encoding="utf-8") as f:
    for linea in f:
        documento = json.loads(linea)
        testo = documento["testo_originale"]
        doc_id = documento["document_id"]

        aperture = len(tag_apertura.findall(testo))
        chiusure = len(tag_chiusura.findall(testo))

        totale_aperture += aperture
        totale_chiusure += chiusure

        if aperture != chiusure:
            documenti_sbilanciati.append(
                (doc_id, aperture, chiusure)
            )

        for nome in documento["nomi_annotati"]:
            if not nome.strip():
                annotazioni_vuote.append(doc_id)

            if "<true_name>" in nome.lower() or "</true_name>" in nome.lower():
                annotazioni_con_tag_interni.append((doc_id, nome))

print(f"Tag di apertura: {totale_aperture}")
print(f"Tag di chiusura: {totale_chiusure}")
print(f"Documenti con conteggi diversi: {len(documenti_sbilanciati)}")
print(f"Annotazioni vuote: {len(annotazioni_vuote)}")
print(f"Annotazioni contenenti tag interni: {len(annotazioni_con_tag_interni)}")

if documenti_sbilanciati:
    print("\nEsempi di documenti da controllare:")
    for elemento in documenti_sbilanciati[:10]:
        print(elemento)
