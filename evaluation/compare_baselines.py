import sys
from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent

GOLDEN_FILE = BASE_DIR / "data" / "golden" / "amazonhelp_golden.csv"
MODEL_FILE = BASE_DIR / "data" / "processed" / "tfidf_intent_model.joblib"

# Allow importing src
sys.path.insert(0, str(BASE_DIR / "src"))

from support_agent import run_agent


def main():

    print("=" * 70)
    print("HIVER BASELINE COMPARISON")
    print("=" * 70)

    # --------------------------------------------------
    # Load golden set
    # --------------------------------------------------

    df = pd.read_csv(
        GOLDEN_FILE,
        dtype=str,
        keep_default_na=False
    )

    y_true = df["gold_intent"]

    print(f"\nGolden examples: {len(df)}")

    # --------------------------------------------------
    # 1. MAJORITY CLASS BASELINE
    # --------------------------------------------------

    majority_class = y_true.value_counts().idxmax()

    majority_predictions = [
        majority_class
    ] * len(df)

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

    print("\n" + "-" * 70)
    print("1. MAJORITY-CLASS BASELINE")
    print("-" * 70)

    print(f"Majority class: {majority_class}")
    print(f"Accuracy:       {majority_accuracy:.4f}")
    print(f"Macro F1:       {majority_macro_f1:.4f}")
    print(f"Weighted F1:    {majority_weighted_f1:.4f}")

    # --------------------------------------------------
    # 2. TF-IDF + LOGISTIC REGRESSION
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("2. TF-IDF + LOGISTIC REGRESSION")
    print("-" * 70)

    print("Loading saved baseline model...")

    model = joblib.load(MODEL_FILE)

    X_text = df["customer_text_clean"]

    baseline_predictions = model.predict(X_text)

    baseline_accuracy = accuracy_score(
        y_true,
        baseline_predictions
    )

    baseline_macro_f1 = f1_score(
        y_true,
        baseline_predictions,
        average="macro",
        zero_division=0
    )

    baseline_weighted_f1 = f1_score(
        y_true,
        baseline_predictions,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy:       {baseline_accuracy:.4f}")
    print(f"Macro F1:       {baseline_macro_f1:.4f}")
    print(f"Weighted F1:    {baseline_weighted_f1:.4f}")

    # --------------------------------------------------
    # 3. HYBRID AI AGENT
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("3. HYBRID AI SUPPORT AGENT")
    print("-" * 70)

    agent_predictions = []

    for i, text in enumerate(X_text):

        try:
            result = run_agent(text)

            prediction = result.get(
                "intent",
                "other"
            )

        except Exception as e:

            print(
                f"Agent error on example {i + 1}: {e}"
            )

            prediction = "other"

        agent_predictions.append(prediction)

        print(
            f"\rProcessed {i + 1}/{len(df)}",
            end=""
        )

    print()

    agent_accuracy = accuracy_score(
        y_true,
        agent_predictions
    )

    agent_macro_f1 = f1_score(
        y_true,
        agent_predictions,
        average="macro",
        zero_division=0
    )

    agent_weighted_f1 = f1_score(
        y_true,
        agent_predictions,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy:       {agent_accuracy:.4f}")
    print(f"Macro F1:       {agent_macro_f1:.4f}")
    print(f"Weighted F1:    {agent_weighted_f1:.4f}")

    # --------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------

    comparison = pd.DataFrame({
        "System": [
            "Majority Class",
            "TF-IDF + Logistic Regression",
            "Hybrid AI Agent"
        ],
        "Accuracy": [
            majority_accuracy,
            baseline_accuracy,
            agent_accuracy
        ],
        "Macro F1": [
            majority_macro_f1,
            baseline_macro_f1,
            agent_macro_f1
        ],
        "Weighted F1": [
            majority_weighted_f1,
            baseline_weighted_f1,
            agent_weighted_f1
        ]
    })

    print("\n")
    print("=" * 70)
    print("FINAL BASELINE COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    output_file = BASE_DIR / "evaluation" / "baseline_comparison.csv"

    comparison.to_csv(
        output_file,
        index=False
    )

    print("\nSaved:")
    print(output_file)


if __name__ == "__main__":
    main()
