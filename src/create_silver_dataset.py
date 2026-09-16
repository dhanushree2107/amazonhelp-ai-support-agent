import pandas as pd
import re
from pathlib import Path


INPUT_FILE = "data/processed/amazonhelp_clean.csv"
OUTPUT_FILE = "data/processed/amazonhelp_silver.csv"

CHUNK_SIZE = 50_000


INTENT_RULES = {
    "delivery_issue": [
        "delivery", "delayed", "delay", "late", "where is my",
        "not arrived", "haven't received", "didn't receive",
        "missing package", "tracking", "delivered but",
        "parcel", "package", "shipment"
    ],

    "return_refund": [
        "refund", "return", "money back", "reimburse",
        "refund hasn't", "refund not"
    ],

    "wrong_damaged_item": [
        "damaged", "damage", "broken", "wrong item",
        "wrong product", "defective", "arrived damaged"
    ],

    "account_access": [
        "can't login", "cannot login", "login", "password",
        "account locked", "locked account", "email address",
        "security", "account access"
    ],

    "payment_billing": [
        "payment", "charged", "charge", "billing",
        "credit card", "debit card", "payment method"
    ],

    "prime_membership": [
        "amazon prime", "prime membership", "prime member",
        "prime subscription", "prime benefits"
    ],

    "technical_issue": [
        "app", "website", "error", "not working",
        "doesn't work", "cannot play", "can't play",
        "fire tv", "kindle", "alexa", "audible",
        "prime video", "technical"
    ],

    "cancellation_change": [
        "cancel order", "cancel my order", "change my order",
        "modify order", "change delivery", "cancel"
    ],

    "order_issue": [
        "place order", "order something", "order number",
        "order status", "my order"
    ],

    "customer_service": [
        "customer service", "support team", "agent",
        "representative", "complaint", "speak to someone",
        "contact support"
    ]
}


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def assign_intent(text):
    text = clean_text(text)

    if not text:
        return "other"

    scores = {}

    for intent, keywords in INTENT_RULES.items():
        score = 0

        for keyword in keywords:
            if keyword in text:
                score += 1

        if score > 0:
            scores[intent] = score

    if not scores:
        return "other"

    # Highest keyword match wins
    return max(scores, key=scores.get)


def main():

    print("=" * 70)
    print("STEP 13.2 - CREATE SILVER TRAINING DATASET")
    print("=" * 70)

    output_chunks = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=False
        ),
        start=1
    ):

        print(f"\nProcessing chunk {chunk_number}...")

        chunk["silver_intent"] = chunk[
            "customer_text_clean"
        ].apply(assign_intent)

        output_chunks.append(
            chunk[
                [
                    "customer_tweet_id",
                    "customer_text_clean",
                    "brand_text",
                    "silver_intent"
                ]
            ]
        )

    result = pd.concat(
        output_chunks,
        ignore_index=True
    )

    Path(OUTPUT_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("SILVER DATASET CREATED")
    print("=" * 70)

    print(f"Rows: {len(result):,}")

    print("\nIntent distribution:")
    print(
        result["silver_intent"]
        .value_counts()
        .to_string()
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
