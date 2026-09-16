import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from support_agent import run_agent


GOLDEN_FILE = BASE_DIR / "data" / "golden" / "amazonhelp_golden.csv"
OUTPUT_FILE = BASE_DIR / "evaluation" / "agent_predictions.csv"
METRICS_FILE = BASE_DIR / "evaluation" / "metrics.txt"


def main():

    print("=" * 70)
    print("HIVER AI SUPPORT AGENT - GOLDEN EVALUATION")
    print("=" * 70)

    df = pd.read_csv(
        GOLDEN_FILE,
        dtype=str,
        keep_default_na=False
    )

    print(f"\nGolden examples: {len(df)}")
    print("\nRunning evaluation...\n")

    predictions = []
    confidences = []
    sources = []
    decisions = []
    reasons = []

    for i, row in df.iterrows():

        text = row["customer_text_clean"]

        try:
            result = run_agent(text)

            predictions.append(
                result.get("intent", "other")
            )

            confidences.append(
                float(result.get("confidence", 0.0))
            )

            sources.append(
                result.get("classification_source", "UNKNOWN")
            )

            decisions.append(
                result.get("decision", "human")
            )

            reasons.append(
                result.get("escalation_reason", "")
            )

            print(
                f"[{i+1:03d}/{len(df)}] "
                f"gold={row['gold_intent']:<22} "
                f"pred={result.get('intent', 'other'):<22} "
                f"conf={float(result.get('confidence', 0.0)):.2f} "
                f"source={result.get('classification_source', 'UNKNOWN')}"
            )

        except Exception as e:

            print(
                f"[{i+1:03d}/{len(df)}] ERROR: {e}"
            )

            predictions.append("other")
            confidences.append(0.0)
            sources.append("ERROR")
            decisions.append("human")
            reasons.append(str(e))

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------

    df["predicted_intent"] = predictions
    df["confidence"] = confidences
    df["classification_source"] = sources
    df["decision"] = decisions
    df["decision_reason"] = reasons

    y_true = df["gold_intent"]
    y_pred = df["predicted_intent"]

    # --------------------------------------------------
    # Classification metrics
    # --------------------------------------------------

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

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    )

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # --------------------------------------------------
    # Decision metrics
    # --------------------------------------------------

    auto_count = (
        df["decision"] == "auto"
    ).sum()

    human_count = (
        df["decision"] == "human"
    ).sum()

    high_conf_wrong = (
        (df["gold_intent"] != df["predicted_intent"])
        & (
            pd.to_numeric(df["confidence"])
            >= 0.90
        )
    ).sum()

    # --------------------------------------------------
    # Classification source
    # --------------------------------------------------

    source_counts = (
        df["classification_source"]
        .value_counts()
    )

    # --------------------------------------------------
    # Errors
    # --------------------------------------------------

    wrong = df[
        df["gold_intent"]
        != df["predicted_intent"]
    ]

    # --------------------------------------------------
    # Final metrics text
    # --------------------------------------------------

    metrics_text = f"""
HIVER AI SUPPORT AGENT
GOLDEN SET EVALUATION

Golden examples: {len(df)}

========================================
HEADLINE CLASSIFICATION RESULTS
========================================

Accuracy:      {accuracy:.4f}
Macro F1:      {macro_f1:.4f}
Weighted F1:   {weighted_f1:.4f}

========================================
ESCALATION RESULTS
========================================

Auto-handle:   {auto_count} ({auto_count / len(df) * 100:.2f}%)
Human:         {human_count} ({human_count / len(df) * 100:.2f}%)

High-confidence wrong predictions (>= 0.90):
{high_conf_wrong}

========================================
CLASSIFICATION SOURCES
========================================

{source_counts.to_string()}

========================================
CLASSIFICATION REPORT
========================================

{report}

========================================
CONFUSION MATRIX LABELS
========================================

{labels}

========================================
CONFUSION MATRIX
========================================

{cm}

========================================
TOTAL ERRORS
========================================

{len(wrong)}
"""

    # --------------------------------------------------
    # Save files
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(metrics_text)

    # --------------------------------------------------
    # Terminal summary
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Accuracy:         {accuracy:.4f}"
    )

    print(
        f"Macro F1:         {macro_f1:.4f}"
    )

    print(
        f"Weighted F1:      {weighted_f1:.4f}"
    )

    print(
        f"Auto-handle:      {auto_count} "
        f"({auto_count / len(df) * 100:.2f}%)"
    )

    print(
        f"Human escalation: {human_count} "
        f"({human_count / len(df) * 100:.2f}%)"
    )

    print(
        f"High-conf wrong:  {high_conf_wrong}"
    )

    print(
        f"Total errors:     {len(wrong)}"
    )

    print("\nSaved:")
    print(f"  {OUTPUT_FILE}")
    print(f"  {METRICS_FILE}")

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(report)


if __name__ == "__main__":
    main()
