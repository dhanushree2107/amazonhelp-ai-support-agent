import pandas as pd
import re
import os

INPUT_FILE = "data/processed/amazonhelp_pairs.csv"
OUTPUT_FILE = "data/processed/amazonhelp_clean.csv"

print("=" * 60)
print("STEP 9 - CLEAN AMAZONHELP DATASET")
print("=" * 60)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original rows: {len(df):,}")

# ---------------------------------------------------------
# 1. REMOVE DUPLICATE CUSTOMER TWEET IDs
# ---------------------------------------------------------

before = len(df)

df = df.drop_duplicates(
    subset=["customer_tweet_id"],
    keep="first"
)

print(
    f"After duplicate tweet IDs removed: "
    f"{len(df):,}"
)

print(
    f"Removed: {before - len(df):,}"
)

# ---------------------------------------------------------
# 2. REMOVE DUPLICATE CUSTOMER TEXT
# ---------------------------------------------------------

before = len(df)

df = df.drop_duplicates(
    subset=["customer_text"],
    keep="first"
)

print(
    f"After duplicate customer texts removed: "
    f"{len(df):,}"
)

print(
    f"Removed: {before - len(df):,}"
)

# ---------------------------------------------------------
# 3. CREATE MODELING TEXT
# ---------------------------------------------------------
# Keep original customer_text unchanged.
# customer_text_clean will be used by the ML model.

def clean_text(text):

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove Twitter mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Convert HTML ampersand artifact
    text = text.replace("&amp;", " and ")

    # Remove excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["customer_text_clean"] = (
    df["customer_text"]
    .apply(clean_text)
)

# ---------------------------------------------------------
# 4. REMOVE EMPTY CLEANED MESSAGES
# ---------------------------------------------------------

before = len(df)

df = df[
    df["customer_text_clean"].str.len() > 2
].copy()

print(
    f"After removing empty messages: "
    f"{len(df):,}"
)

print(
    f"Removed: {before - len(df):,}"
)

# ---------------------------------------------------------
# 5. BASIC QUALITY CHECK
# ---------------------------------------------------------

print("\nMissing values:")

print(
    df[
        [
            "customer_tweet_id",
            "customer_text",
            "customer_text_clean",
            "brand_text"
        ]
    ].isnull().sum()
)

print("\nClean dataset shape:")
print(df.shape)

# ---------------------------------------------------------
# 6. SAVE
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)

# ---------------------------------------------------------
# 7. SHOW EXAMPLES
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("CLEANING EXAMPLES")
print("-" * 60)

examples = df.sample(
    n=min(10, len(df)),
    random_state=42
)

for i, row in enumerate(
    examples.itertuples(index=False),
    start=1
):

    print(f"\n{i}. ORIGINAL:")
    print(row.customer_text)

    print("   CLEANED:")
    print(row.customer_text_clean)

print("\n" + "=" * 60)
print("STEP 9 COMPLETE")
print("=" * 60)
