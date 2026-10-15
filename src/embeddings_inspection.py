'''
Lo script controlla la struttura dei file contenenti gli embedding delle strategie 
baseline M0–M3 e della strategia LLM M4. Legge il primo record disponibile di ciascun 
file JSONL e stampa informazioni sui metodi presenti, sui tipi di dato, sulle chiavi 
dei record e sulle dimensioni dei vettori, mostrando anche alcuni valori iniziali. 
Ha una funzione esclusivamente diagnostica, utile per verificare il formato degli 
output prima delle successive elaborazioni. Non calcola metriche sperimentali e non 
modifica i file.
'''

import json
from pathlib import Path

FILES = [
    Path("data/processed/embeddings_100.jsonl"),
    Path("data/processed/embeddings_m4_100.jsonl"),
]

for path in FILES:
    if not path.exists():
        print(f"\nFile non trovato: {path}")
        continue

    print(f"\n{'=' * 60}\n{path}")

    with path.open("r", encoding="utf-8") as f:
        record = json.loads(next(line for line in f if line.strip()))

    for method, chunks in record["embeddings"].items():
        print(f"\nMetodo: {method}")
        print("Tipo:", type(chunks).__name__)

        if isinstance(chunks, list):
            print("Numero elementi:", len(chunks))

            if chunks:
                first = chunks[0]
                print("Tipo primo elemento:", type(first).__name__)

                if isinstance(first, list):
                    print("Dimensione primo vettore:", len(first))
                    print("Primi 5 valori:", first[:5])
                else:
                    print("Primo elemento:", str(first)[:150])

        elif isinstance(chunks, dict):
            print("Chiavi:", list(chunks.keys())[:10])
            for key, value in list(chunks.items())[:2]:
                print(
                    f"  {key}: tipo={type(value).__name__}, "
                    f"lunghezza={len(value) if hasattr(value, '__len__') else 'n/d'}"
                )
        else:
            print("Valore:", str(chunks)[:150])
