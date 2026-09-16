import pandas as pd
from pathlib import Path

INPUT_FILE = "data/processed/amazonhelp_clean.csv"
OUTPUT_FILE = "data/golden/golden_candidates.csv"

RANDOM_STATE = 42
SAMPLES_PER_INTENT = 18

# Candidate keywords are ONLY used to find examples.
# They are NOT treated as ground truth.
INTENT_KEYWORDS = {
    "delivery_issue": [
        "delivery", "delivered", "late", "package", "parcel",
        "shipping", "tracking", "courier"
    ],

    "order_issue": [
        "order", "ordering", "ordered", "purchase"
    ],

    "return_refund": [
        "refund", "return", "money back", "reimburse"
    ],

    "wrong_damaged_item": [
        "wrong item", "damaged", "broken", "empty box",
        "incorrect item", "missing item"
    ],

    "payment_billing": [
        "charged", "charge", "payment", "billing", "card",
        "debit", "credit", "fee"
    ],

    "prime_membership": [
        "prime", "membership", "subscription"
    ],

    "account_access": [
        "account", "login", "log in", "locked", "password"
    ],

    "technical_issue": [
        "fire tv", "kindle", "echo", "alexa",
        "prime video", "buffering", "app", "device"
    ],

    "cancellation_change": [
        "cancel", "cancellation", "change my order",
        "re-order", "reorder"
    ],

    "customer_service": [
        "customer service", "support", "agent",
        "representative", "contact you", "speak to"
    ],
}


def find_candidates(text, keywords):
    text = str(text).lower()
    return any(keyword in text for keyword in keywords)


def main():
    print("=" * 60)
    print("STEP 12 - GOLDEN SET CANDIDATE SAMPLING")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded messages: {len(df):,}")

    candidates = []

    for intent, keywords in INTENT_KEYWORDS.items():

        mask = df["customer_text_clean"].apply(
            lambda x: find_candidates(x, keywords)
        )

        subset = df[mask].copy()

        if len(subset) > SAMPLES_PER_INTENT:
            subset = subset.sample(
                n=SAMPLES_PER_INTENT,
                random_state=RANDOM_STATE
            )

        subset["candidate_intent"] = intent

        candidates.append(subset)

        print(
            f"{intent:25s} -> "
            f"{len(subset):3d} candidates"
        )

    golden = pd.concat(candidates, ignore_index=True)

    # Remove duplicate customer tweets
    golden = golden.drop_duplicates(
        subset=["customer_tweet_id"]
    )

    # Keep only columns needed for manual labelling
    golden = golden[
        [
            "customer_tweet_id",
            "customer_text",
            "customer_text_clean",
            "brand_text",
            "candidate_intent"
        ]
    ]

    # Add manual ground-truth column
    golden["gold_intent"] = ""

    # Add notes for difficult examples
    golden["label_notes"] = ""

    # Shuffle so candidate intents aren't grouped together
    golden = golden.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    Path(OUTPUT_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    golden.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Golden candidates created: {len(golden)}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    print("IMPORTANT:")
    print("candidate_intent = discovery suggestion only")
    print("gold_intent = manually verified ground truth")


if __name__ == "__main__":
    main()
