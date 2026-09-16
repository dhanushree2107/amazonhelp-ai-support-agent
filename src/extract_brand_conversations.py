import pandas as pd

DATA_PATH = "data/twcs.csv"

BRAND = "AmazonHelp"

USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id"
]

print("=" * 70)
print(f"EXTRACTING {BRAND} CONVERSATIONS")
print("=" * 70)

# ---------------------------------------------------------
# STEP 1: Load all tweets written by the selected brand
# ---------------------------------------------------------

brand_rows = []

print("\nScanning dataset for brand tweets...")

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=100000,
    usecols=USECOLS
):

    rows = chunk[chunk["author_id"] == BRAND]

    if len(rows) > 0:
        brand_rows.append(rows)

brand_df = pd.concat(brand_rows, ignore_index=True)

print("Brand tweets found:", len(brand_df))


# ---------------------------------------------------------
# STEP 2: Collect IDs connected to brand tweets
# ---------------------------------------------------------

related_ids = set()

for _, row in brand_df.iterrows():

    # Previous tweet
    if pd.notna(row["in_response_to_tweet_id"]):
        related_ids.add(
            str(row["in_response_to_tweet_id"]).replace(".0", "")
        )

    # Next tweets
    if pd.notna(row["response_tweet_id"]):

        response_ids = str(
            row["response_tweet_id"]
        ).split(",")

        for tweet_id in response_ids:
            related_ids.add(tweet_id.strip())


print("Related tweet IDs:", len(related_ids))


# ---------------------------------------------------------
# STEP 3: Find those tweets in the full dataset
# ---------------------------------------------------------

related_rows = []

print("\nFinding related tweets...")

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=100000,
    usecols=USECOLS
):

    chunk["tweet_id_str"] = (
        chunk["tweet_id"]
        .astype(str)
        .str.replace(".0", "", regex=False)
    )

    matches = chunk[
        chunk["tweet_id_str"].isin(related_ids)
    ]

    if len(matches) > 0:
        related_rows.append(matches)

related_df = pd.concat(
    related_rows,
    ignore_index=True
)

print("Related tweets found:", len(related_df))


# ---------------------------------------------------------
# STEP 4: Combine brand + related tweets
# ---------------------------------------------------------

conversation_df = pd.concat(
    [
        brand_df,
        related_df
    ],
    ignore_index=True
)

conversation_df = conversation_df.drop_duplicates(
    subset=["tweet_id"]
)

print(
    "Total conversation tweets:",
    len(conversation_df)
)


# ---------------------------------------------------------
# STEP 5: Show real examples
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE CONVERSATION TWEETS")
print("=" * 70)

conversation_df = conversation_df.sort_values(
    "tweet_id"
)

for _, row in conversation_df.head(30).iterrows():

    print("\n" + "-" * 70)

    print("Tweet ID:", row["tweet_id"])
    print("Author:", row["author_id"])
    print("Inbound:", row["inbound"])
    print("Created:", row["created_at"])

    print("Text:")
    print(row["text"])

    print("In Response To:", row["in_response_to_tweet_id"])
    print("Response Tweet:", row["response_tweet_id"])
