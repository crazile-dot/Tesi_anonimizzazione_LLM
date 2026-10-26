'''
valuta quanto bene il modello OpenAI individua i nomi di persona e quanto riesce effettivamente a sostituirli nei documenti.
'''

import json
import re
from pathlib import Path
from collections import Counter


# =============================================================
# PATH
# =============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "openai_results_100.jsonl"

# =============================================================
# UTILITY
# =============================================================

def normalize(text):
    """
    Normalizza una stringa per i confronti:
    - rimuove spazi iniziali/finali
    - converte in lowercase
    - comprime gli spazi multipli
    """
    if not text:
        return ""

    return re.sub(r"\s+", " ", text.strip().lower())


def count_occurrences(text, phrase):
    """
    Conta le occorrenze non sovrapposte di phrase nel testo.
    Il confronto è case-insensitive.
    """
    if not phrase:
        return 0

    return len(
        re.findall(
            re.escape(phrase),
            text,
            flags=re.IGNORECASE
        )
    )


def calculate_metrics(tp, fp, fn):
    """
    Calcola Precision, Recall e F1.
    """

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else None
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else None
    )

    if (
        precision is not None
        and recall is not None
        and precision + recall > 0
    ):
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = None

    return precision, recall, f1


# =============================================================
# ENTITY-LEVEL
# =============================================================

def evaluate_entity_level(record):
    """
    Valuta la capacità del modello di identificare
    le entità PERSON presenti nel ground truth.

    Ogni entità viene considerata una sola volta,
    indipendentemente dal numero di occorrenze nel documento.
    """

    ground_truth = record.get("ground_truth", [])
    entities_detected = record.get("entities_detected", [])

    # ---------------------------------------------------------
    # GOLD
    # ---------------------------------------------------------

    gold_entities = set()

    for entity in ground_truth:

        name = normalize(
            entity.get("text", "")
        )

        if name:
            gold_entities.add(name)

    # ---------------------------------------------------------
    # PREDICTED
    # ---------------------------------------------------------

    predicted_entities = set()

    for entity in entities_detected:

        name = normalize(
            entity.get("original", "")
        )

        if name:
            predicted_entities.add(name)

    # ---------------------------------------------------------
    # TP / FP / FN
    # ---------------------------------------------------------

    true_positive = len(
        gold_entities & predicted_entities
    )

    false_positive = len(
        predicted_entities - gold_entities
    )

    false_negative = len(
        gold_entities - predicted_entities
    )

    return (
        true_positive,
        false_positive,
        false_negative
    )


# =============================================================
# MENTION-LEVEL
# =============================================================

def evaluate_mention_level(record):
    """
    Valuta quante occorrenze PERSON vengono effettivamente
    protette nell'output anonimizzato.

    La valutazione è basata sul conteggio delle occorrenze
    e non sulla posizione esatta degli span.

    Più varianti dello stesso nome possono essere associate
    allo stesso placeholder. In questo caso le occorrenze del
    placeholder vengono conteggiate una sola volta.
    """

    anonymized_text = record.get(
        "text_anonymized",
        ""
    )

    ground_truth = record.get(
        "ground_truth",
        []
    )

    entities_detected = record.get(
        "entities_detected",
        []
    )

    # ---------------------------------------------------------
    # GOLD MENTIONS
    # ---------------------------------------------------------

    gold_counts = Counter()

    for entity in ground_truth:

        name = normalize(
            entity.get("text", "")
        )

        if name:
            gold_counts[name] += 1

    # ---------------------------------------------------------
    # PREDICTED ENTITIES
    # ---------------------------------------------------------

    # Mappa:
    # nome originale normalizzato -> replacement
    predicted_entities = {}

    # Raggruppa le varianti per replacement:
    #
    # PERSON_1 -> {
    #     "brian t. hoskins",
    #     "brian hoskins"
    # }
    #
    # PERSON_2 -> {
    #     "john house"
    # }
    replacement_groups = {}

    invalid_replacements = []

    for entity in entities_detected:

        original = entity.get(
            "original",
            ""
        )

        replacement = entity.get(
            "replacement",
            ""
        )

        original_norm = normalize(original)

        if not original_norm or not replacement:
            continue

        predicted_entities[original_norm] = replacement

        replacement_groups.setdefault(
            replacement,
            set()
        ).add(original_norm)

    # ---------------------------------------------------------
    # CONTROLLO COERENZA REPLACEMENT
    # ---------------------------------------------------------

    for replacement in replacement_groups:

        replacement_occurrences = count_occurrences(
            anonymized_text,
            replacement
        )

        if replacement_occurrences == 0:

            for original_norm in replacement_groups[replacement]:

                invalid_replacements.append({
                    "original": original_norm,
                    "replacement": replacement
                })

    # ---------------------------------------------------------
    # TP / FN
    # ---------------------------------------------------------

    true_positive = 0
    false_negative = 0

    # Teniamo traccia dei replacement che sono associati
    # ad almeno un'entità presente nel ground truth.
    gold_replacements = set()

    for replacement, predicted_names in replacement_groups.items():

        # -----------------------------------------------------
        # Quali nomi di questo gruppo sono presenti nel gold?
        # -----------------------------------------------------

        matching_gold_names = (
            predicted_names
            & set(gold_counts.keys())
        )

        # Nessun nome del gruppo è presente nel ground truth.
        # Questo replacement verrà gestito come FP più avanti.
        if not matching_gold_names:
            continue

        gold_replacements.add(replacement)

        # -----------------------------------------------------
        # Numero totale di mention gold associate
        # a questo replacement
        # -----------------------------------------------------

        gold_count_for_replacement = sum(
            gold_counts[name]
            for name in matching_gold_names
        )

        # -----------------------------------------------------
        # Numero di volte in cui il placeholder compare
        # nell'output anonimizzato
        # -----------------------------------------------------

        predicted_count = count_occurrences(
            anonymized_text,
            replacement
        )

        # -----------------------------------------------------
        # TP / FN
        #
        # IMPORTANTE:
        # il replacement viene contato UNA SOLA VOLTA,
        # anche se più varianti del nome sono associate
        # allo stesso placeholder.
        # -----------------------------------------------------

        correct = min(
            gold_count_for_replacement,
            predicted_count
        )

        true_positive += correct

        false_negative += (
            gold_count_for_replacement - correct
        )

    # ---------------------------------------------------------
    # FN PER ENTITÀ GOLD NON RILEVATE
    # ---------------------------------------------------------

    for gold_name, gold_count in gold_counts.items():

        # Il nome non è stato rilevato dal modello.
        if gold_name not in predicted_entities:

            false_negative += gold_count

    # ---------------------------------------------------------
    # FP
    # ---------------------------------------------------------

    false_positive = 0

    for replacement, predicted_names in replacement_groups.items():

        # Se il replacement è associato ad almeno un'entità
        # gold, le sue occorrenze sono già state considerate
        # nel calcolo TP/FN.
        if replacement in gold_replacements:
            continue

        # Altrimenti tutte le occorrenze del replacement
        # appartengono a entità non presenti nel ground truth.
        predicted_count = count_occurrences(
            anonymized_text,
            replacement
        )

        false_positive += predicted_count

    return (
        true_positive,
        false_positive,
        false_negative,
        invalid_replacements
    )

# =============================================================
# MAIN EVALUATION
# =============================================================

def evaluate_m4(filepath):

    print("\n" + "=" * 60)
    print("M4 - LLM SUBSTITUTION")
    print("=" * 60)

    if not filepath.exists():

        print(
            f"\nFile non trovato:\n{filepath}"
        )

        return

    # ---------------------------------------------------------
    # CONTATORI DOCUMENTI
    # ---------------------------------------------------------

    total_documents = 0
    successful_documents = 0
    skipped_documents = 0

    # ---------------------------------------------------------
    # ENTITY-LEVEL MICRO
    # ---------------------------------------------------------

    entity_tp = 0
    entity_fp = 0
    entity_fn = 0

    # ---------------------------------------------------------
    # MENTION-LEVEL MICRO
    # ---------------------------------------------------------

    mention_tp = 0
    mention_fp = 0
    mention_fn = 0

    # ---------------------------------------------------------
    # MACRO
    # ---------------------------------------------------------

    entity_document_metrics = []
    mention_document_metrics = []

    # ---------------------------------------------------------
    # ERRORI / INCOERENZE
    # ---------------------------------------------------------

    documents_with_invalid_replacements = 0

    # ---------------------------------------------------------
    # LETTURA
    # ---------------------------------------------------------

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            total_documents += 1

            record = json.loads(line)

            # -------------------------------------------------
            # DOCUMENTO NON ELABORATO
            # -------------------------------------------------

            if "text_anonymized" not in record:

                skipped_documents += 1
                continue

            successful_documents += 1

            # -------------------------------------------------
            # ENTITY-LEVEL
            # -------------------------------------------------

            tp, fp, fn = evaluate_entity_level(
                record
            )

            entity_tp += tp
            entity_fp += fp
            entity_fn += fn

            p, r, f1 = calculate_metrics(
                tp,
                fp,
                fn
            )

            if (
                p is not None
                and r is not None
                and f1 is not None
            ):

                entity_document_metrics.append(
                    (p, r, f1)
                )

            # -------------------------------------------------
            # MENTION-LEVEL
            # -------------------------------------------------

            tp, fp, fn, invalid = evaluate_mention_level(
                record
            )

            mention_tp += tp
            mention_fp += fp
            mention_fn += fn

            if invalid:
                documents_with_invalid_replacements += 1

            p, r, f1 = calculate_metrics(
                tp,
                fp,
                fn
            )

            if (
                p is not None
                and r is not None
                and f1 is not None
            ):

                mention_document_metrics.append(
                    (p, r, f1)
                )

    # =========================================================
    # MICRO METRICS
    # =========================================================

    entity_micro_p, entity_micro_r, entity_micro_f1 = (
        calculate_metrics(
            entity_tp,
            entity_fp,
            entity_fn
        )
    )

    mention_micro_p, mention_micro_r, mention_micro_f1 = (
        calculate_metrics(
            mention_tp,
            mention_fp,
            mention_fn
        )
    )

    # =========================================================
    # MACRO METRICS
    # =========================================================

    if entity_document_metrics:

        entity_macro_p = sum(
            x[0] for x in entity_document_metrics
        ) / len(entity_document_metrics)

        entity_macro_r = sum(
            x[1] for x in entity_document_metrics
        ) / len(entity_document_metrics)

        entity_macro_f1 = sum(
            x[2] for x in entity_document_metrics
        ) / len(entity_document_metrics)

    else:

        entity_macro_p = None
        entity_macro_r = None
        entity_macro_f1 = None

    if mention_document_metrics:

        mention_macro_p = sum(
            x[0] for x in mention_document_metrics
        ) / len(mention_document_metrics)

        mention_macro_r = sum(
            x[1] for x in mention_document_metrics
        ) / len(mention_document_metrics)

        mention_macro_f1 = sum(
            x[2] for x in mention_document_metrics
        ) / len(mention_document_metrics)

    else:

        mention_macro_p = None
        mention_macro_r = None
        mention_macro_f1 = None

    # =========================================================
    # OUTPUT
    # =========================================================

    print(
        f"\nDocumenti totali:       "
        f"{total_documents}"
    )

    print(
        f"Documenti validi:       "
        f"{successful_documents}"
    )

    print(
        f"Documenti ignorati:     "
        f"{skipped_documents}"
    )

    print(
        f"Documenti con replacement "
        f"incoerenti:            "
        f"{documents_with_invalid_replacements}"
    )

    # =========================================================
    # ENTITY-LEVEL
    # =========================================================

    print("\n" + "-" * 60)
    print("ENTITY-LEVEL")
    print("-" * 60)

    print("\nMICRO")

    print(
        f"TP: {entity_tp} | "
        f"FP: {entity_fp} | "
        f"FN: {entity_fn}"
    )

    if entity_micro_p is not None:

        print(
            f"Precision: {entity_micro_p:.4f}"
        )

        print(
            f"Recall:    {entity_micro_r:.4f}"
        )

        print(
            f"F1:        {entity_micro_f1:.4f}"
        )

    else:

        print("Metriche non disponibili.")

    print("\nMACRO")

    if entity_macro_p is not None:

        print(
            f"Precision: {entity_macro_p:.4f}"
        )

        print(
            f"Recall:    {entity_macro_r:.4f}"
        )

        print(
            f"F1:        {entity_macro_f1:.4f}"
        )

    else:

        print("Metriche non disponibili.")

    # =========================================================
    # MENTION-LEVEL
    # =========================================================

    print("\n" + "-" * 60)
    print("MENTION-LEVEL")
    print("-" * 60)

    print("\nMICRO")

    print(
        f"TP: {mention_tp} | "
        f"FP: {mention_fp} | "
        f"FN: {mention_fn}"
    )

    if mention_micro_p is not None:

        print(
            f"Precision: {mention_micro_p:.4f}"
        )

        print(
            f"Recall:    {mention_micro_r:.4f}"
        )

        print(
            f"F1:        {mention_micro_f1:.4f}"
        )

    else:

        print("Metriche non disponibili.")

    print("\nMACRO")

    if mention_macro_p is not None:

        print(
            f"Precision: {mention_macro_p:.4f}"
        )

        print(
            f"Recall:    {mention_macro_r:.4f}"
        )

        print(
            f"F1:        {mention_macro_f1:.4f}"
        )

    else:

        print("Metriche non disponibili.")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    evaluate_m4(INPUT_FILE)