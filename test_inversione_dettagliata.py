
import json
import time
from pathlib import Path

import torch
import vec2text
from openai import OpenAI


MODEL = "text-embedding-ada-002"
OUTPUT_DIR = Path("risultati_diagnostica")
OUTPUT_DIR.mkdir(exist_ok=True)

TESTI = [
    "Alice met Bob at the library on Tuesday.",
    "The finance team will review the quarterly report next Monday.",
    "Maria sent the revised contract to James after the meeting in London.",
]


def stampa_separatore():
    print("\n" + "=" * 90)


def mostra_embedding(embedding, indice):
    vettore = embedding.detach().cpu().tolist()

    print(f"\nEMBEDDING {indice}")
    print("-" * 50)
    print(f"Tipo tensor: {embedding.dtype}")
    print(f"Forma tensor: {tuple(embedding.shape)}")
    print(f"Numero componenti: {len(vettore)}")
    print(f"Prime 10 componenti: {vettore[:10]}")
    print(f"Ultime 10 componenti: {vettore[-10:]}")
    print(f"Valore minimo: {min(vettore):.8f}")
    print(f"Valore massimo: {max(vettore):.8f}")
    print(f"Norma L2: {torch.linalg.vector_norm(embedding).item():.8f}")

    # Salva il vettore completo, senza stamparlo tutto a video.
    with open(
        OUTPUT_DIR / f"embedding_{indice}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(vettore, file)

    print(f"Vettore completo salvato in embedding_{indice}.json")


def main():
    print("FASE 1 - TESTI ORIGINALI")
    stampa_separatore()

    for i, testo in enumerate(TESTI, 1):
        print(f"Testo {i}: {testo}")

    # Verifica che la chiave sia presente senza mostrarla.
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY assente. Imposta la variabile d'ambiente."
        )

    client = OpenAI()

    print("\nFASE 2 - GENERAZIONE DEGLI EMBEDDING")
    print(f"Modello: {MODEL}")
    print("Invio dei tre testi all'API...", flush=True)

    risposta = client.embeddings.create(
        model=MODEL,
        input=TESTI,
        encoding_format="float",
    )

    # Ordina le risposte secondo l'indice restituito dall'API.
    dati = sorted(risposta.data, key=lambda item: item.index)

    embeddings = torch.tensor(
        [elemento.embedding for elemento in dati],
        dtype=torch.float32,
    )

    print(f"\nModello dichiarato nella risposta: {risposta.model}")
    print(f"Forma del batch: {tuple(embeddings.shape)}")

    # Conserva anche tutti i vettori insieme in un unico file.
    torch.save(embeddings, OUTPUT_DIR / "embeddings_completi.pt")

    for i, (testo, embedding) in enumerate(zip(TESTI, embeddings), 1):
        stampa_separatore()
        print(f"TEST {i}")
        print("ORIGINALE:", testo)
        mostra_embedding(embedding, i)

    print("\nFASE 3 - CARICAMENTO DEL MODELLO DI INVERSIONE")
    print("Caricamento del corrector Vec2Text...", flush=True)

    corrector = vec2text.load_pretrained_corrector(MODEL)

    print("Corrector caricato.")

    risultati = {}

    for num_steps in (0, 5):
        stampa_separatore()
        print(f"FASE 4 - INVERSIONE CON num_steps={num_steps}")
        print("Il processo può richiedere tempo su CPU...", flush=True)

        inizio = time.time()

        ricostruzioni = vec2text.invert_embeddings(
            embeddings=embeddings,
            corrector=corrector,
            num_steps=num_steps,
        )

        durata = time.time() - inizio
        risultati[str(num_steps)] = []

        print(f"Tempo impiegato: {durata:.2f} secondi")

        for i, (originale, ricostruito) in enumerate(
            zip(TESTI, ricostruzioni), 1
        ):
            identico = originale == ricostruito

            record = {
                "test": i,
                "originale": originale,
                "ricostruito": ricostruito,
                "identico": identico,
                "num_steps": num_steps,
            }
            risultati[str(num_steps)].append(record)

            print(f"\n--- TEST {i} ---")
            print("ORIGINALE:   ", originale)
            print("RICOSTRUITO: ", repr(ricostruito))
            print("IDENTICO:    ", identico)

    stampa_separatore()
    print("FASE 5 - CONFRONTO FINALE")

    for i, originale in enumerate(TESTI, 1):
        print(f"\nTest {i}: {originale}")

        for steps in ("0", "5"):
            ricostruito = risultati[steps][i - 1]["ricostruito"]
            print(f"  steps={steps}: {ricostruito}")

    with open(
        OUTPUT_DIR / "confronto_ricostruzioni.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            risultati,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nRisultati salvati in:")
    print(OUTPUT_DIR / "confronto_ricostruzioni.json")
    print("\nDiagnostica completata.")


if __name__ == "__main__":
    main()

