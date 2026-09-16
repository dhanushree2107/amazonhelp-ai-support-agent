import pandas as pd

INPUT_FILE = "evaluation/agent_predictions.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("HIGH-CONFIDENCE FAILURE ANALYSIS")
print("=" * 70)

# Wrong predictions
wrong = df[df["gold_intent"] != df["predicted_intent"]].copy()

print(f"\nTotal examples: {len(df)}")
print(f"Wrong predictions: {len(wrong)}")
print(f"Accuracy: {(len(df) - len(wrong)) / len(df):.2%}")

# High-confidence wrong predictions
high_conf = wrong[wrong["confidence"] >= 0.85].copy()

print(f"\nHigh-confidence wrong predictions: {len(high_conf)}")

print("\n" + "=" * 70)
print("TOP CONFIDENT FAILURES")
print("=" * 70)

for _, row in high_conf.sort_values(
    "confidence", ascending=False
).head(40).iterrows():

    print("\n" + "-" * 70)
    print(f"Example ID: {row.get('example_id', '')}")
    print(f"Gold intent: {row['gold_intent']}")
    print(f"Predicted:   {row['predicted_intent']}")
    print(f"Confidence:  {row['confidence']:.2f}")
    print(f"Source:      {row.get('classification_source', '')}")
    print(f"\nCustomer message:\n{row['customer_text']}")

# Confusion pairs
print("\n" + "=" * 70)
print("TOP CONFUSION PAIRS")
print("=" * 70)

confusions = (
    wrong.groupby(
        ["gold_intent", "predicted_intent"]
    )
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(confusions.head(20).to_string(index=False))

# Errors by classification source
print("\n" + "=" * 70)
print("ERRORS BY CLASSIFICATION SOURCE")
print("=" * 70)

if "classification_source" in wrong.columns:
    print(
        wrong["classification_source"]
        .value_counts()
        .to_string()
    )

# Errors by predicted intent
print("\n" + "=" * 70)
print("MOST COMMON WRONG PREDICTIONS")
print("=" * 70)

print(
    wrong["predicted_intent"]
    .value_counts()
    .head(15)
    .to_string()
)

print("\nAnalysis complete.")
