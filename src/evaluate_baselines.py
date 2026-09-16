# ============================================================
# BASELINE EVALUATION
# Majority Class vs TF-IDF + Logistic Regression
# ============================================================

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# ------------------------------------------------------------
# FILES
# ------------------------------------------------------------

GOLDEN_FILE = "data/golden/amazonhelp_golden.csv"

MODEL_FILE = "data/processed/tfidf_intent_model_v2.joblib"


# ------------------------------------------------------------
# LOAD GOLDEN DATA
# ------------------------------------------------------------

print("Loading golden evaluation set...")

df = pd.read_csv(
    GOLDEN_FILE,
    dtype=str,
    keep_default_na=False
)

df = df[
    df["gold_intent"].str.strip() != ""
].copy()

print(
    f"Human-labelled examples: {len(df)}"
)


if len(df) == 0:
    raise ValueError(
        "No human-labelled examples found."
    )


# ------------------------------------------------------------
# GOLD LABELS
# ------------------------------------------------------------

y_true = df["gold_intent"]


# ------------------------------------------------------------
# BASELINE 1: MAJORITY CLASS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BASELINE 1: MAJORITY CLASS")
print("=" * 60)

majority_class = (
    y_true.value_counts()
    .idxmax()
)

majority_predictions = [
    majority_class
] * len(y_true)

majority_accuracy = accuracy_score(
    y_true,
    majority_predictions
)

majority_macro_f1 = f1_score(
    y_true,
    majority_predictions,
    average="macro",
    zero_division=0
)

majority_weighted_f1 = f1_score(
    y_true,
    majority_predictions,
    average="weighted",
    zero_division=0
)

print(
    f"Majority class: {majority_class}"
)

print(
    f"Accuracy: {majority_accuracy:.4f}"
)

print(
    f"Macro F1: {majority_macro_f1:.4f}"
)

print(
    f"Weighted F1: {majority_weighted_f1:.4f}"
)


# ------------------------------------------------------------
# BASELINE 2: TF-IDF + LOGISTIC REGRESSION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BASELINE 2: TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

print("Loading trained model...")

model = joblib.load(
    MODEL_FILE
)


# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

texts = df["customer_text_clean"]

predictions = model.predict(
    texts
)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

tfidf_accuracy = accuracy_score(
    y_true,
    predictions
)

tfidf_macro_f1 = f1_score(
    y_true,
    predictions,
    average="macro",
    zero_division=0
)

tfidf_weighted_f1 = f1_score(
    y_true,
    predictions,
    average="weighted",
    zero_division=0
)


print(
    f"Accuracy: {tfidf_accuracy:.4f}"
)

print(
    f"Macro F1: {tfidf_macro_f1:.4f}"
)

print(
    f"Weighted F1: {tfidf_weighted_f1:.4f}"
)


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        predictions,
        zero_division=0
    )
)


# ------------------------------------------------------------
# COMPARISON
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)

print(
    f"{'Model':35s} "
    f"{'Accuracy':>10s} "
    f"{'Macro F1':>10s} "
    f"{'Weighted F1':>12s}"
)

print("-" * 70)

print(
    f"{'Majority Class':35s} "
    f"{majority_accuracy:>10.4f} "
    f"{majority_macro_f1:>10.4f} "
    f"{majority_weighted_f1:>12.4f}"
)

print(
    f"{'TF-IDF + Logistic Regression':35s} "
    f"{tfidf_accuracy:>10.4f} "
    f"{tfidf_macro_f1:>10.4f} "
    f"{tfidf_weighted_f1:>12.4f}"
)

print("=" * 60)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results = pd.DataFrame([
    {
        "model": "Majority Class",
        "accuracy": majority_accuracy,
        "macro_f1": majority_macro_f1,
        "weighted_f1": majority_weighted_f1
    },
    {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": tfidf_accuracy,
        "macro_f1": tfidf_macro_f1,
        "weighted_f1": tfidf_weighted_f1
    }
])

results.to_csv(
    "evaluation/baseline_results.csv",
    index=False
)

print(
    "\nSaved results to "
    "evaluation/baseline_results.csv"
)
