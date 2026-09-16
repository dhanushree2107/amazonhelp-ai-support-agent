import pandas as pd
from pathlib import Path

INPUT_FILE = "data/golden/golden_candidates_200.csv"
OUTPUT_FILE = "data/golden/amazonhelp_golden.csv"

INTENTS = {
    "1": "delivery_issue",
    "2": "order_issue",
    "3": "return_refund",
    "4": "wrong_damaged_item",
    "5": "payment_billing",
    "6": "prime_membership",
    "7": "account_access",
    "8": "technical_issue",
    "9": "cancellation_change",
    "10": "customer_service",
    "11": "other",
}


def load_csv(path):
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

if Path(OUTPUT_FILE).exists():
    df = load_csv(OUTPUT_FILE)
    print("Continuing existing labeling progress.")
else:
    df = load_csv(INPUT_FILE)
    print("Starting new labeling session.")

# Make sure columns exist
if "gold_intent" not in df.columns:
    df["gold_intent"] = ""

if "label_notes" not in df.columns:
    df["label_notes"] = ""

df["gold_intent"] = df["gold_intent"].fillna("").astype(str)
df["label_notes"] = df["label_notes"].fillna("").astype(str)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

BATCH_SIZE = 10


# ------------------------------------------------------------
# DISPLAY INTENTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FAST GOLDEN SET LABELING")
print("=" * 70)

print("""
1  = delivery_issue
2  = order_issue
3  = return_refund
4  = wrong_damaged_item
5  = payment_billing
6  = prime_membership
7  = account_access
8  = technical_issue
9  = cancellation_change
10 = customer_service
11 = other
q  = quit and save
""")


# ------------------------------------------------------------
# LABELING LOOP
# ------------------------------------------------------------

while True:

    unlabeled = df[
        df["gold_intent"].str.strip() == ""
    ]

    if len(unlabeled) == 0:
        print("\n🎉 All examples have been labelled!")
        break

    batch = unlabeled.head(BATCH_SIZE)

    print("\n" + "=" * 70)
    print(
        f"Showing {len(batch)} examples "
        f"| Remaining: {len(unlabeled)}"
    )
    print("=" * 70)

    for position, (idx, row) in enumerate(batch.iterrows(), start=1):

        print("\n" + "-" * 70)
        print(f"EXAMPLE {position}/{len(batch)}")
        print(f"Row index: {idx}")
        print("-" * 70)

        print("\nCUSTOMER:")
        print(row["customer_text"])

        print("\nHISTORICAL AMAZON REPLY:")
        print(row["brand_text"])

        print("\nCANDIDATE SUGGESTION:")
        print(row.get("candidate_intent", ""))

        print("-" * 70)

    # --------------------------------------------------------
    # GET LABELS
    # --------------------------------------------------------

    print("\nEnter labels in order.")
    print("Example:")
    print("1,1,7,3,10,9,1,8,2,1")

    print("\nYou can also enter:")
    print("q = quit and save")

    answer = input("\nLabels: ").strip()

    if answer.lower() == "q":
        df.to_csv(
            OUTPUT_FILE,
            index=False
        )
        print("\nProgress saved.")
        break

    # --------------------------------------------------------
    # PARSE LABELS
    # --------------------------------------------------------

    labels = [
        x.strip()
        for x in answer.split(",")
        if x.strip()
    ]

    if len(labels) != len(batch):
        print(
            f"\n❌ Expected {len(batch)} labels "
            f"but received {len(labels)}."
        )
        print("Please enter exactly one label per example.")
        continue

    invalid = [
        x for x in labels
        if x not in INTENTS
    ]

    if invalid:
        print(
            "\n❌ Invalid labels:",
            invalid
        )
        print(
            "Valid labels are 1 to 11."
        )
        continue

    # --------------------------------------------------------
    # SAVE LABELS
    # --------------------------------------------------------

    for (idx, row), label in zip(
        batch.iterrows(),
        labels
    ):

        intent = INTENTS[label]

        df.at[idx, "gold_intent"] = intent

        print(
            f"Example {idx}: "
            f"{label} -> {intent}"
        )

    # --------------------------------------------------------
    # OPTIONAL NOTES
    # --------------------------------------------------------

    print(
        "\nLabels saved for this batch."
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Progress saved to: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    remaining = len(
        df[
            df["gold_intent"].str.strip() == ""
        ]
    )

    labelled = len(df) - remaining

    print("\n" + "=" * 70)
    print(
        f"Progress: {labelled}/{len(df)} labelled"
    )
    print(
        f"Remaining: {remaining}"
    )
    print("=" * 70)


print("\nDone.")
