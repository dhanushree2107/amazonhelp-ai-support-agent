import pandas as pd
import joblib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


GOLDEN_FILE = "data/golden/amazonhelp_golden.csv"
MODEL_FILE = "data/processed/tfidf_intent_model_v2.joblib"


def main():

    print("=" * 70)
    print("STEP 20 - GOLDEN SET EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load golden dataset
    # ---------------------------------------------------------

    print("\nLoading golden evaluation set...")

    df = pd.read_csv(
        GOLDEN_FILE,
        dtype=str,
        keep_default_na=False
    )

    # Keep only genuinely manually labelled examples

    df = df[
        df["gold_intent"].str.strip() != ""
    ].copy()

    print(
        f"Human-labelled examples: {len(df)}"
    )

    if len(df) == 0:

        print("No human-labelled examples found.")

        return

    # ---------------------------------------------------------
    # Load trained classifier
    # ---------------------------------------------------------

    print("\nLoading classifier...")

    model = joblib.load(
        MODEL_FILE
    )

    print("Classifier loaded.")

    # ---------------------------------------------------------
    # Prepare input
    # ---------------------------------------------------------

    X = df["customer_text_clean"]
    y_true = df["gold_intent"]

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    print("\nRunning predictions...")

    y_pred = model.predict(X)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("GOLDEN SET RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print("\nConfusion Matrix:")

    print(
        pd.DataFrame(
            cm,
            index=labels,
            columns=labels
        )
    )

    # ---------------------------------------------------------
    # Individual predictions
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("INDIVIDUAL PREDICTIONS")
    print("=" * 70)

    df["predicted_intent"] = y_pred

    for index, row in df.iterrows():

        correct = (
            row["gold_intent"]
            == row["predicted_intent"]
        )

        status = "CORRECT" if correct else "WRONG"

        print(
            f"\n[{status}] Example {index + 1}"
        )

        print(
            f"Customer: {row['customer_text']}"
        )

        print(
            f"Gold: {row['gold_intent']}"
        )

        print(
            f"Predicted: {row['predicted_intent']}"
        )

        if row["label_notes"].strip():

            print(
                f"Label note: {row['label_notes']}"
            )


if __name__ == "__main__":
    main()
