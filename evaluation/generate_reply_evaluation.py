import sys
import json
import pandas as pd

sys.path.insert(0, "src")

from support_agent import run_agent


INPUT_FILE = "data/golden/amazonhelp_golden.csv"
OUTPUT_FILE = "evaluation/reply_evaluation.csv"


def main():

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    results = []

    print("=" * 70)
    print("REPLY EVALUATION DATASET GENERATOR")
    print("=" * 70)
    print(f"Examples: {len(df)}")
    print()

    for i, row in df.iterrows():

        print("-" * 70)
        print(f"Example {i + 1}/{len(df)}")
        print(f"Customer: {row['customer_text'][:180]}")

        try:

            result = run_agent(row["customer_text"])

            evidence = result.get("evidence", [])

            # Save top 3 historical examples as JSON
            evidence_json = json.dumps(
                evidence[:3],
                ensure_ascii=False
            )

            results.append({
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": row["customer_text"],
                "gold_intent": row["gold_intent"],

                "predicted_intent": result.get(
                    "intent",
                    ""
                ),

                "confidence": result.get(
                    "confidence",
                    ""
                ),

                "classification_source": result.get(
                    "classification_source",
                    ""
                ),

                "decision": result.get(
                    "decision",
                    ""
                ),

                "decision_reason": result.get(
                    "escalation_reason",
                    result.get("decision_reason", "")
                ),

                "reply": result.get(
                    "reply",
                    ""
                ),

                "evidence": evidence_json
            })

            print(
                f"Intent: {result.get('intent', '')}"
            )

            print(
                f"Decision: {result.get('decision', '')}"
            )

            print(
                f"Reply: {result.get('reply', '')[:180]}"
            )

        except Exception as e:

            print(f"ERROR: {e}")

            results.append({
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": row["customer_text"],
                "gold_intent": row["gold_intent"],
                "predicted_intent": "",
                "confidence": "",
                "classification_source": "",
                "decision": "",
                "decision_reason": "",
                "reply": "",
                "evidence": "",
            })

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Rows saved: {len(output_df)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
