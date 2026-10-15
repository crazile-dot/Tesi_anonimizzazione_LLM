'''
Lo script confronta gli embedding dei testi originali (M0) con quelli ottenuti applicando 
le tre strategie di anonimizzazione baseline: redazione (M1), pseudonimizzazione 
deterministica (M2) e HMAC (M3). A partire dal file embeddings_100.jsonl, calcola 
la distanza euclidea tra i vettori di M0 e quelli di ciascuna strategia, limitatamente 
a cinque documenti selezionati manualmente. I risultati vengono stampati a terminale e consentono 
un'analisi preliminare delle variazioni delle rappresentazioni semantiche introdotte dall'anonimizzazione. 
Lo script ha finalità esplorative: non produce statistiche sull'intero dataset e non valuta direttamente 
né il rischio di inversione degli embedding né le prestazioni del semantic retrieval.
'''
import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "embeddings_100.jsonl"

TARGET_DOCUMENTS = {
    "baughman-e_all_documents_13",
    "campbell-l_all_documents_867",
    "hain-m_all_documents_448",
    "whalley-g_all_documents_595",
    "wolfe-j_all_documents_91"
}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        document = json.loads(line)

        if document["document_id"] not in TARGET_DOCUMENTS:
            continue

        embeddings = document["embeddings"]

        print("Numero chunk M0:", embeddings["M0_original"]["num_chunks"])

        m0 = np.array(embeddings["M0_original"]["embedding"])

        print("----------------------------------------")
        print(document["document_id"])
        print("----------------------------------------")

        for method in ["M1_redaction", "M2_pseudonymization", "M3_hmac"]:
            vector = np.array(embeddings[method]["embedding"])
            distance = np.linalg.norm(m0 - vector)

            print(f"M0 - {method}: {distance:.17f}")
