import pandas as pd

DATA_PATH = "data/twcs.csv"
BRAND = "AmazonHelp"

print(f"Checking quality of {BRAND} conversations...\n")

samples = []

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=100_000,
    dtype={
        "tweet_id": "Int64",
        "author_id": "string",
        "inbound": "boolean",
        "created_at": "string",
        "text": "string",
        "response_tweet_id": "string",
        "in_response_to_tweet_id": "string",
    },
):
    # AmazonHelp tweets
    brand_rows = chunk[chunk["author_id"] == BRAND]

    # Customer tweets
    customer_rows = chunk[chunk["inbound"] == True]

    # Collect some examples
    if len(samples) < 50:
        for _, row in brand_rows.head(20).iterrows():
            samples.append(
                {
                    "type": "BRAND",
                    "tweet_id": row["tweet_id"],
                    "text": row["text"],
                }
            )

    if len(samples) < 50:
        for _, row in customer_rows.head(30).iterrows():
            samples.append(
                {
                    "type": "CUSTOMER",
                    "tweet_id": row["tweet_id"],
                    "text": row["text"],
                }
            )

    if len(samples) >= 50:
        break


df = pd.DataFrame(samples)

print("=" * 70)
print("SAMPLE CONVERSATION QUALITY")
print("=" * 70)

for _, row in df.iterrows():
    print(f"\n[{row['type']}] Tweet ID: {row['tweet_id']}")
    print(row["text"])

print("\n" + "=" * 70)

# Basic quality statistics from the sampled data
print("QUALITY CHECK")
print("=" * 70)

print(f"Sample size: {len(df)}")
print(f"Empty texts: {df['text'].isna().sum()}")
print(f"Duplicate texts: {df['text'].duplicated().sum()}")

if len(df) > 0:
    lengths = df["text"].fillna("").str.len()

    print(f"Average text length: {lengths.mean():.1f}")
    print(f"Shortest text: {lengths.min()}")
    print(f"Longest text: {lengths.max()}")

print("\nQuality check completed.")