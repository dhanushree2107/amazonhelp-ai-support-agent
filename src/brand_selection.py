import pandas as pd
from collections import Counter

DATA_PATH = "data/twcs.csv"

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "British_Airways",
    "Tesco",
    "comcastcares",
    "sainsburys",
    "Delta",
    "VirginTrains",
]

print("Loading dataset in chunks...")

brand_tweets = {brand: set() for brand in BRANDS}
brand_counts = Counter()

# --------------------------------------------------
# PASS 1: Collect tweet IDs belonging to each brand
# --------------------------------------------------

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=100_000,
    dtype={
        "tweet_id": "Int64",
        "author_id": "string",
        "inbound": "boolean",
        "text": "string",
        "response_tweet_id": "string",
        "in_response_to_tweet_id": "string",
    },
):
    for brand in BRANDS:
        rows = chunk[chunk["author_id"] == brand]

        brand_counts[brand] += len(rows)

        brand_tweets[brand].update(
            rows["tweet_id"].dropna().astype(str)
        )

print("\nBrand tweet counts:")
for brand, count in brand_counts.most_common():
    print(f"{brand:20} {count:,}")


# --------------------------------------------------
# PASS 2: Find customer messages directly answered
#         by each brand
# --------------------------------------------------

customer_pairs = Counter()

print("\nFinding customer → brand conversations...")

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=100_000,
    dtype={
        "tweet_id": "Int64",
        "author_id": "string",
        "inbound": "boolean",
        "text": "string",
        "response_tweet_id": "string",
        "in_response_to_tweet_id": "string",
    },
):
    customers = chunk[chunk["inbound"] == True].copy()

    for _, row in customers.iterrows():

        response_ids = row["response_tweet_id"]

        if pd.isna(response_ids):
            continue

        response_ids = str(response_ids)

        # response_tweet_id can contain multiple IDs
        ids = {
            x.strip()
            for x in response_ids.split(",")
            if x.strip()
        }

        for brand in BRANDS:
            if ids.intersection(brand_tweets[brand]):
                customer_pairs[brand] += 1


# --------------------------------------------------
# FINAL COMPARISON
# --------------------------------------------------

print("\n" + "=" * 65)
print("BRAND SELECTION RESULTS")
print("=" * 65)

print(
    f"{'Brand':20} {'Brand Tweets':>15} {'Answered Customers':>20}"
)

print("-" * 65)

for brand, _ in brand_counts.most_common():

    print(
        f"{brand:20} "
        f"{brand_counts[brand]:>15,} "
        f"{customer_pairs[brand]:>20,}"
    )

print("=" * 65)

print("\nRecommended candidates:")
for brand, count in customer_pairs.most_common(5):
    print(f"{brand:20} {count:,} answered customer messages")