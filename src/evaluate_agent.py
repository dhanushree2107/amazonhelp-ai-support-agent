
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# ============================================================
# Windows Unicode output
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

# ============================================================
# Allow imports from src/
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from support_agent import (
    classify_message,
    retrieve_similar,
    decide_escalation,
)


# ============================================================
# File paths
# ============================================================

GOLDEN_FILE = "data/golden/amazonhelp_golden.csv"

OUTPUT_FILE = "evaluation/agent_predictions.csv"

SUMMARY_FILE = "evaluation/agent_summary.csv"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("COMPLETE AMAZON SUPPORT AI AGENT EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load golden evaluation set
    # --------------------------------------------------------

    df = pd.read_csv(
        GOLDEN_FILE,
        dtype=str,
        keep_default_na=False
    )

    df = df[
        df["gold_intent"].str.strip() != ""
    ].copy()

    print(f"\nGolden examples: {len(df)}")

    results = []

    # ========================================================
    # Evaluate all examples
    # ========================================================

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        customer_text = row["customer_text"]

        print(
            f"\rEvaluating example "
            f"{position}/{len(df)}...",
            end="",
            flush=True
        )

        try:

            # ------------------------------------------------
            # 1. Actual deployed classifier
            # ------------------------------------------------
            #
            # classify_message() returns a DICTIONARY:
            #
            # {
            #     "intent": ...,
            #     "confidence": ...,
            #     "source": ...
            # }
            #
            # So we must access the dictionary keys.
            # ------------------------------------------------

            classification = classify_message(
                customer_text
            )

            predicted_intent = classification[
                "intent"
            ]

            confidence = float(
                classification[
                    "confidence"
                ]
            )

            source = classification[
                "source"
            ]

            # ------------------------------------------------
            # 2. Historical retrieval
            # ------------------------------------------------

            evidence = retrieve_similar(
                customer_text,
                predicted_intent,
                top_k=3
            )

            # ------------------------------------------------
            # 3. Escalation policy
            # ------------------------------------------------

            escalation = decide_escalation(
                customer_text,
                classification,
                evidence
                )
            decision = escalation["decision"]
            reason = escalation["reason"]
            # ------------------------------------------------
            # 4. Top retrieval similarity
            # ------------------------------------------------

            top_similarity = 0.0

            if evidence:

                try:

                    top_similarity = float(
                        evidence[0]["similarity"]
                    )

                except (
                    TypeError,
                    ValueError,
                    KeyError
                ):

                    top_similarity = 0.0

            # ------------------------------------------------
            # 5. Store result
            # ------------------------------------------------

            results.append({

                "example_id": position,

                "customer_text": customer_text,

                "gold_intent": row[
                    "gold_intent"
                ],

                "predicted_intent":
                    predicted_intent,

                "confidence":
                    confidence,

                "classification_source":
                    source,

                "top_similarity":
                    top_similarity,

                "decision":
                    decision,

                "escalation_reason":
                    reason

            })

        except Exception as e:

            print(
                f"\nWARNING: Example "
                f"{position} failed: {e}"
            )

            results.append({

                "example_id": position,

                "customer_text": customer_text,

                "gold_intent": row[
                    "gold_intent"
                ],

                "predicted_intent":
                    "ERROR",

                "confidence":
                    0.0,

                "classification_source":
                    "ERROR",

                "top_similarity":
                    0.0,

                "decision":
                    "ERROR",

                "escalation_reason":
                    str(e)

            })

    print("\n")

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # INTENT CLASSIFICATION
    # ========================================================

    valid_results = results_df[
        results_df[
            "predicted_intent"
        ] != "ERROR"
    ].copy()

    y_true = valid_results[
        "gold_intent"
    ]

    y_pred = valid_results[
        "predicted_intent"
    ]

    if len(valid_results) > 0:

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

    else:

        accuracy = 0.0
        macro_f1 = 0.0
        weighted_f1 = 0.0

    # ========================================================
    # RESULTS
    # ========================================================

    print("=" * 70)
    print("INTENT CLASSIFICATION RESULTS")
    print("=" * 70)

    print(
        f"\nValid examples: "
        f"{len(valid_results)}"
    )

    print(
        f"Failed examples: "
        f"{len(results_df) - len(valid_results)}"
    )

    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{weighted_f1:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    if len(valid_results) > 0:

        print(
            classification_report(
                y_true,
                y_pred,
                zero_division=0
            )
        )

    # ========================================================
    # RULE VS ML
    # ========================================================

    print("=" * 70)
    print("CLASSIFICATION SOURCE")
    print("=" * 70)

    print(
        results_df[
            "classification_source"
        ].value_counts()
    )

    # ========================================================
    # ESCALATION
    # ========================================================

    print("\n" + "=" * 70)
    print("ESCALATION RESULTS")
    print("=" * 70)

    valid_decisions = results_df[
        results_df[
            "decision"
        ] != "ERROR"
    ].copy()

    decision_counts = (
        valid_decisions[
            "decision"
        ]
        .value_counts()
    )

    print(
        "\nDecision distribution:"
    )

    print(
        decision_counts
    )

    total = len(
        valid_decisions
    )

    auto_handled = (
        valid_decisions[
            "decision"
        ]
        == "AUTO_HANDLE"
    ).sum()

    escalated = (
        valid_decisions[
            "decision"
        ]
        == "ESCALATE_TO_HUMAN"
    ).sum()

    if total > 0:

        auto_handle_rate = (
            auto_handled / total
        )

        escalation_rate = (
            escalated / total
        )

    else:

        auto_handle_rate = 0.0
        escalation_rate = 0.0

    print(
        f"\nTotal valid examples: "
        f"{total}"
    )

    print(
        f"Auto-handled: "
        f"{auto_handled}"
    )

    print(
        f"Escalated: "
        f"{escalated}"
    )

    print(
        f"Auto-handle rate: "
        f"{auto_handle_rate:.2%}"
    )

    print(
        f"Escalation rate: "
        f"{escalation_rate:.2%}"
    )

    # ========================================================
    # ESCALATION BY INTENT
    # ========================================================

    print("\n" + "=" * 70)
    print("ESCALATION BY INTENT")
    print("=" * 70)

    if len(valid_decisions) > 0:

        escalation_by_intent = pd.crosstab(
            valid_decisions[
                "gold_intent"
            ],
            valid_decisions[
                "decision"
            ]
        )

        print(
            escalation_by_intent
        )

    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    print("\n" + "=" * 70)
    print("LOW CONFIDENCE ANALYSIS")
    print("=" * 70)

    low_conf = results_df[
        results_df[
            "confidence"
        ] < 0.55
    ]

    print(
        f"Examples below confidence "
        f"threshold: {len(low_conf)}"
    )

    # ========================================================
    # LOW RETRIEVAL EVIDENCE
    # ========================================================

    print("\n" + "=" * 70)
    print("LOW RETRIEVAL EVIDENCE")
    print("=" * 70)

    low_retrieval = results_df[
        results_df[
            "top_similarity"
        ] < 0.20
    ]

    print(
        f"Examples with similarity < 0.20: "
        f"{len(low_retrieval)}"
    )

    # ========================================================
    # HIGH-CONFIDENCE WRONG PREDICTIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE MISCLASSIFICATIONS")
    print("=" * 70)

    high_conf_wrong = results_df[
        (results_df[
            "gold_intent"
        ]
        != results_df[
            "predicted_intent"
        ])
        &
        (results_df[
            "predicted_intent"
        ] != "ERROR")
        &
        (results_df[
            "confidence"
        ] >= 0.80)
    ]

    print(
        f"High-confidence wrong predictions: "
        f"{len(high_conf_wrong)}"
    )

    if len(high_conf_wrong) > 0:

        display_columns = [
            "example_id",
            "gold_intent",
            "predicted_intent",
            "confidence",
            "customer_text"
        ]

        print(
            high_conf_wrong[
                display_columns
            ]
            .head(10)
            .to_string(index=False)
        )

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    Path(
        "evaluation"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\nDetailed predictions:"
        f"\n{OUTPUT_FILE}"
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = pd.DataFrame([{

        "examples":
            len(results_df),

        "valid_examples":
            len(valid_results),

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "auto_handled":
            auto_handled,

        "escalated":
            escalated,

        "auto_handle_rate":
            auto_handle_rate,

        "escalation_rate":
            escalation_rate,

        "low_confidence_examples":
            len(low_conf),

        "low_retrieval_examples":
            len(low_retrieval),

        "high_confidence_wrong":
            len(high_conf_wrong)

    }])

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(
        f"\nSummary:"
        f"\n{SUMMARY_FILE}"
    )

    # ========================================================
    # HEADLINE RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("HEADLINE RESULTS")
    print("=" * 70)

    print(
        f"\nIntent Accuracy : "
        f"{accuracy:.2%}"
    )

    print(
        f"Macro F1       : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1    : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Auto-handle    : "
        f"{auto_handle_rate:.2%}"
    )

    print(
        f"Escalation     : "
        f"{escalation_rate:.2%}"
    )

    print(
        "\nEvaluation complete."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
