import pandas as pd
from pathlib import Path

INPUT_FILE = "data/processed/amazonhelp_clean.csv"
GOLDEN_FILE = "data/golden/golden_candidates.csv"
OUTPUT_FILE = "data/golden/golden_candidates_200.csv"

RANDOM_STATE = 42
OTHER_COUNT = 20

INTENT_KEYWORDS = [
    "delivery",
    "delivered",
    "late",
    "package",
    "parcel",
    "shipping",
    "tracking",
    "order",
    "ordering",
    "ordered",
    "purchase",
    "refund",
    "return",
    "damaged",
    "broken",
    "wrong item",
    "incorrect",
    "charge",
    "charged",
    "payment",
    "billing",
    "card",
    "prime",
    "membership",
    "subscription",
    "account",
    "login",
    "log in",
    "password",
    "locked",
    "fire tv",
    "kindle",
    "echo",
    "alexa",
    "prime video",
    "buffering",
    "cancel",
    "cancellation",
    "customer service",
    "support",
    "agent",
    "representative",
]


def looks_like_other(text):
    text = str(text).lower()

    return not any(
        keyword in text
        for keyword in INTENT_KEYWORDS
    )


def main():

    print("=" * 60)
    print("STEP 12.2 - ADD OTHER INTENT")
    print("=" * 60)

    golden = pd.read_csv(GOLDEN_FILE)
    df = pd.read_csv(INPUT_FILE)

    print(f"Existing golden candidates: {len(golden)}")

    # Find messages that don't strongly match
    # our discovery keywords.
    mask = df["customer_text_clean"].apply(
        looks_like_other
    )

    other_pool = df[mask].copy()

    # Avoid duplicate customer tweets
    other_pool = other_pool[
        ~other_pool["customer_tweet_id"].isin(
            golden["customer_tweet_id"]
        )
    ]

    print(f"Potential 'other' pool: {len(other_pool)}")

    other = other_pool.sample(
        n=OTHER_COUNT,
        random_state=RANDOM_STATE
    )

    other["candidate_intent"] = "other"
    other["gold_intent"] = ""
    other["label_notes"] = ""

    other = other[
        [
            "customer_tweet_id",
            "customer_text",
            "customer_text_clean",
            "brand_text",
            "candidate_intent",
            "gold_intent",
            "label_notes"
        ]
    ]

    golden = golden[
        [
            "customer_tweet_id",
            "customer_text",
            "customer_text_clean",
            "brand_text",
            "candidate_intent",
            "gold_intent",
            "label_notes"
        ]
    ]

    final = pd.concat(
        [golden, other],
        ignore_index=True
    )

    final = final.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    Path(OUTPUT_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Final golden candidates: {len(final)}")
    print(f"Output: {OUTPUT_FILE}")

    print()
    print("Candidate distribution:")
    print(final["candidate_intent"].value_counts())

    print()
    print("Next step:")
    print("Manually fill the gold_intent column.")


if __name__ == "__main__":
    main()
