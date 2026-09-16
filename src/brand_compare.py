import pandas as pd

DATA_PATH = "data/twcs.csv"

brands = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "British_Airways"
]

print("=" * 70)
print("SCANNING DATASET FOR BRAND CONVERSATIONS")
print("=" * 70)

# ---------------------------------------------------------
# STEP 1: Find all tweet IDs belonging to our candidate brands
# ---------------------------------------------------------

brand_tweet_ids = {brand: set() for brand in brands}

print("\nCollecting brand tweet IDs...")

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=100000,
    usecols=["tweet_id", "author_id", "inbound"]
):

    for brand in brands:
        ids = chunk.loc[
            chunk["author_id"] == brand,
            "tweet_id"
        ].astype(str)

        brand_tweet_ids[brand].update(ids)

print("\nBrand tweet counts:")

for brand in brands:
    print(
        f"{brand}: {len(brand_tweet_ids[brand])}"
    )


# ---------------------------------------------------------
# STEP 2: Scan entire dataset and find customer replies
# ---------------------------------------------------------

results = {
    brand: {
        "customer_replies": 0,
        "unique_customers": set(),
        "customer_replies_with_response": 0
    }
    for brand in brands
}

print("\nScanning for customer replies...")

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=100000,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
):

    chunk["parent_id"] = (
        chunk["in_response_to_tweet_id"]
        .fillna("")
        .astype(str)
    )

    for brand in brands:

        matching = chunk[
            (chunk["inbound"] == True) &
            (chunk["parent_id"].isin(brand_tweet_ids[brand]))
        ]

        if len(matching) > 0:

            results[brand]["customer_replies"] += len(matching)

            results[brand]["unique_customers"].update(
                matching["author_id"].astype(str)
            )

            results[brand]["customer_replies_with_response"] += (
                matching["response_tweet_id"].notna().sum()
            )


# ---------------------------------------------------------
# STEP 3: Print results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL BRAND COMPARISON")
print("=" * 70)

for brand in brands:

    print(f"\nBrand: {brand}")
    print("-" * 50)

    print(
        "Brand tweets:",
        len(brand_tweet_ids[brand])
    )

    print(
        "Customer replies to brand:",
        results[brand]["customer_replies"]
    )

    print(
        "Unique customers:",
        len(results[brand]["unique_customers"])
    )

    print(
        "Customer replies with response:",
        results[brand]["customer_replies_with_response"]
    )