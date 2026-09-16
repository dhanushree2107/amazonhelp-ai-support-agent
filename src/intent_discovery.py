import pandas as pd
import re

INPUT_FILE = "data/processed/amazonhelp_clean.csv"

print("=" * 60)
print("STEP 10 - INTENT DISCOVERY")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"\nLoaded {len(df):,} messages")

# ---------------------------------------------------------
# Candidate intents
# ---------------------------------------------------------

INTENTS = {
    "delivery_issue": [
        "late", "delay", "delayed", "delivery", "delivered",
        "not arrived", "didn't arrive", "did not arrive",
        "missing package", "missing parcel", "tracking",
        "where is my order", "where's my order", "out for delivery"
    ],

    "return_refund": [
        "refund", "refunded", "return", "returned",
        "money back", "reimbursement", "refund pending"
    ],

    "wrong_damaged_item": [
        "wrong item", "wrong product", "damaged", "broken",
        "defective", "empty package", "empty box",
        "missing item", "incorrect item", "received wrong"
    ],

    "account_access": [
        "account", "login", "logged in", "log in",
        "password", "locked", "verification", "otp",
        "two step", "2-step", "cannot access"
    ],

    "payment_billing": [
        "payment", "paid", "charge", "charged", "billing",
        "card", "credit card", "debit", "transaction",
        "bank", "unlawful payment", "unauthorized payment"
    ],

    "prime_membership": [
        "prime", "membership", "subscription",
        "prime member", "prime membership",
        "free trial", "prime video"
    ],

    "technical_issue": [
        "error", "not working", "doesn't work",
        "doesnt work", "won't work", "wont work",
        "app", "website", "server", "fire tv",
        "firetv", "kindle", "alexa", "echo",
        "buffering", "streaming", "video"
    ],

    "cancellation_change": [
        "cancel", "cancellation", "change order",
        "change address", "modify order",
        "cancel order"
    ],

    "customer_service": [
        "customer service", "customer care",
        "support", "representative", "agent",
        "call me", "contact", "complaint",
        "no response", "escalate", "resolution"
    ],

    "order_issue": [
        "order", "ordered", "ordering",
        "order status", "order number",
        "order id", "purchase"
    ]
}


# ---------------------------------------------------------
# Find representative examples
# ---------------------------------------------------------

for intent, keywords in INTENTS.items():

    print("\n" + "=" * 60)
    print(f"INTENT: {intent}")
    print("=" * 60)

    # Build regex
    pattern = "|".join(
        re.escape(keyword)
        for keyword in keywords
    )

    mask = df["customer_text_clean"].str.contains(
        pattern,
        case=False,
        regex=True,
        na=False
    )

    matches = df[mask]

    print(f"Matching messages: {len(matches):,}")

    # Show up to 8 examples
    if len(matches) > 0:

        examples = matches.sample(
            n=min(8, len(matches)),
            random_state=42
        )

        for i, text in enumerate(
            examples["customer_text_clean"],
            start=1
        ):
            print(f"\n{i}. {text}")


# ---------------------------------------------------------
# Messages that match multiple candidate intents
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MULTI-INTENT ANALYSIS")
print("=" * 60)

intent_matches = []

for _, row in df.iterrows():

    text = str(row["customer_text_clean"]).lower()

    matched_intents = []

    for intent, keywords in INTENTS.items():

        for keyword in keywords:

            if keyword.lower() in text:
                matched_intents.append(intent)
                break

    if len(matched_intents) >= 2:

        intent_matches.append({
            "text": row["customer_text_clean"],
            "intents": ", ".join(matched_intents)
        })

print(
    f"Messages matching 2+ candidate intents: "
    f"{len(intent_matches):,}"
)

if intent_matches:

    multi_df = pd.DataFrame(intent_matches)

    print("\nExamples:")

    for _, row in multi_df.head(15).iterrows():

        print(f"\nMessage: {row['text']}")
        print(f"Candidate intents: {row['intents']}")


print("\n" + "=" * 60)
print("STEP 10 COMPLETE")
print("=" * 60)