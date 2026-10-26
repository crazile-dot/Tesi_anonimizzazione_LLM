
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "processed" / "utility_cosine_per_document.csv"

OUTPUT_SUMMARY = (
    BASE_DIR / "data" / "processed" / "utility_cosine_common_subset_summary.csv"
)
OUTPUT_TESTS = (
    BASE_DIR / "data" / "processed" / "utility_cosine_pairwise_tests.csv"
)

STRATEGIES = [
    "M1_redaction",
    "M2_pseudonymization",
    "M3_hmac",
    "M4_llm_substitution",
    "M5_llm_anonymization",
]

BOOTSTRAP_ITERATIONS = 10_000
RANDOM_SEED = 42


# ============================================================
# FUNZIONI
# ============================================================

def bootstrap_mean_ci(
    differences,
    iterations=BOOTSTRAP_ITERATIONS,
    seed=RANDOM_SEED,
):
    """Intervallo di confidenza percentile al 95% della differenza media."""
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(iterations)

    # Elaborazione a blocchi per contenere l'uso della memoria.
    batch_size = 500

    for start in range(0, iterations, batch_size):
        end = min(start + batch_size, iterations)
        indices = rng.integers(
            0, len(values), size=(end - start, len(values))
        )
        bootstrap_means[start:end] = values[indices].mean(axis=1)

    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def holm_correction(p_values):
    """Correzione di Holm per confronti multipli."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)

    running_max = 0.0

    for rank, original_index in enumerate(order):
        corrected = (n - rank) * p_values[original_index]
        running_max = max(running_max, corrected)
        adjusted[original_index] = min(running_max, 1.0)

    return adjusted


# ============================================================
# LETTURA E VALIDAZIONE
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File non trovato: {INPUT_FILE}\n"
        "Verifica che evaluate_semantic_similarity.py sia stato eseguito."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = ["document_id"] + STRATEGIES
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(
        f"Colonne mancanti nel CSV: {missing_columns}\n"
        f"Colonne disponibili: {list(df.columns)}"
    )

if df["document_id"].duplicated().any():
    duplicated = df.loc[
        df["document_id"].duplicated(), "document_id"
    ].tolist()
    raise ValueError(f"document_id duplicati: {duplicated[:10]}")

for strategy in STRATEGIES:
    df[strategy] = pd.to_numeric(df[strategy], errors="coerce")

# Tutte le strategie devono essere confrontate sugli stessi documenti.
common = df.dropna(subset=STRATEGIES).copy()

if common.empty:
    raise ValueError("Nessun documento completo per tutte le strategie.")

values = common[STRATEGIES].to_numpy(dtype=float)

if not np.isfinite(values).all():
    raise ValueError("Sono presenti valori non finiti nelle similarità.")

if ((values < -1.000001) | (values > 1.000001)).any():
    raise ValueError("Sono presenti similarità coseno fuori dall'intervallo [-1, 1].")

print(f"Documenti nel file originale: {len(df)}")
print(f"Documenti completi per tutte le strategie: {len(common)}")


# ============================================================
# 1. STATISTICHE DESCRITTIVE
# ============================================================

summary_rows = []

for strategy in STRATEGIES:
    s = common[strategy]

    summary_rows.append({
        "strategy": strategy,
        "n": int(s.count()),
        "mean": s.mean(),
        "variance_sample": s.var(ddof=1),
        "std_sample": s.std(ddof=1),
        "min": s.min(),
        "Q1": s.quantile(0.25),
        "median_Q2": s.quantile(0.50),
        "Q3": s.quantile(0.75),
        "max": s.max(),
        "IQR": s.quantile(0.75) - s.quantile(0.25),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, float_format="%.8f")

print("\n=== STATISTICHE DESCRITTIVE ===")
print(summary.round(4).to_string(index=False))
print(f"\nSalvato: {OUTPUT_SUMMARY}")


# ============================================================
# 2. CONFRONTI APPAIATI TRA TUTTE LE COPPIE
# ============================================================

test_rows = []

for strategy_a, strategy_b in combinations(STRATEGIES, 2):
    # Differenza positiva: A ha similarità maggiore di B.
    differences = (
        common[strategy_a].to_numpy()
        - common[strategy_b].to_numpy()
    )

    mean_diff = float(np.mean(differences))
    median_diff = float(np.median(differences))
    std_diff = float(np.std(differences, ddof=1))

    ci_low, ci_high = bootstrap_mean_ci(differences)

    # Il test è bilaterale: verifica una differenza, senza
    # presupporre quale strategia debba avere il valore maggiore.
    if np.allclose(differences, 0):
        statistic = 0.0
        p_value = 1.0
    else:
        result = wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    test_rows.append({
        "strategy_A": strategy_a,
        "strategy_B": strategy_b,
        "n_pairs": len(differences),
        "mean_difference_A_minus_B": mean_diff,
        "median_difference_A_minus_B": median_diff,
        "std_difference": std_diff,
        "CI95_mean_diff_lower": ci_low,
        "CI95_mean_diff_upper": ci_high,
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    })

tests = pd.DataFrame(test_rows)

# Correzione di Holm sulle 10 comparazioni.
tests["p_value_holm"] = holm_correction(tests["p_value"].to_numpy())
tests["significant_holm_0.05"] = tests["p_value_holm"] < 0.05

tests.to_csv(OUTPUT_TESTS, index=False, float_format="%.8f")

print("\n=== CONFRONTI APPAIATI: TEST DI WILCOXON ===")
display_columns = [
    "strategy_A",
    "strategy_B",
    "mean_difference_A_minus_B",
    "CI95_mean_diff_lower",
    "CI95_mean_diff_upper",
    "p_value",
    "p_value_holm",
    "significant_holm_0.05",
]
print(tests[display_columns].round(5).to_string(index=False))

print(f"\nSalvato: {OUTPUT_TESTS}")
print("\nAnalisi completata.")
