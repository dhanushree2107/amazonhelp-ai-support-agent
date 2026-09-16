import pandas as pd
import os

INPUT_FILE = "data/twcs.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "amazonhelp_pairs.csv")

BRAND = "AmazonHelp"
CHUNK_SIZE = 100_000

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("STEP 7 - BUILD AMAZONHELP CONVERSATION DATASET")
print("=" * 60)

# ---------------------------------------------------------
# PASS 1
# Collect all AmazonHelp tweet IDs and their information
# ---------------------------------------------------------

print("\n[1/2] Collecting AmazonHelp tweets...")

brand_ids = set()
brand_data = {}

total_brand = 0

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    brand_rows = chunk[
        (chunk["author_id"] == BRAND) &
        (chunk["inbound"] == False)
    ]

    for row in brand_rows.itertuples(index=False):
        tweet_id = str(row.tweet_id)

        brand_ids.add(tweet_id)

        brand_data[tweet_id] = {
            "brand_text": str(row.text),
            "brand_created_at": row.created_at
        }

        total_brand += 1

print(f"AmazonHelp tweets found: {total_brand:,}")
print(f"Unique AmazonHelp tweet IDs: {len(brand_ids):,}")


# ---------------------------------------------------------
# PASS 2
# Find customer tweets whose response points to AmazonHelp
# ---------------------------------------------------------

print("\n[2/2] Extracting customer -> AmazonHelp reply pairs...")

first_write = True
total_pairs = 0
seen_pairs = set()

columns = [
    "customer_tweet_id",
    "customer_created_at",
    "customer_text",
    "brand_tweet_id",
    "brand_created_at",
    "brand_text"
]

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    # Customer messages only
    customers = chunk[
        (chunk["inbound"] == True) &
        (chunk["response_tweet_id"].notna())
    ].copy()

    if customers.empty:
        continue

    # Convert response IDs into individual IDs
    customers["response_tweet_id"] = (
        customers["response_tweet_id"]
        .astype(str)
        .str.split(",")
    )

    customers = customers.explode("response_tweet_id")

    customers["response_tweet_id"] = (
        customers["response_tweet_id"]
        .astype(str)
        .str.strip()
    )

    # Keep only tweets answered by AmazonHelp
    matched = customers[
        customers["response_tweet_id"].isin(brand_ids)
    ].copy()

    if matched.empty:
        continue

    rows = []

    for row in matched.itertuples(index=False):

        customer_id = str(row.tweet_id)
        brand_id = str(row.response_tweet_id)

        pair_key = (customer_id, brand_id)

        # Avoid duplicate relationships
        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)

        brand_info = brand_data.get(brand_id)

        if brand_info is None:
            continue

        rows.append({
            "customer_tweet_id": customer_id,
            "customer_created_at": row.created_at,
            "customer_text": str(row.text),
            "brand_tweet_id": brand_id,
            "brand_created_at": brand_info["brand_created_at"],
            "brand_text": brand_info["brand_text"]
        })

    if rows:

        output_df = pd.DataFrame(rows, columns=columns)

        output_df.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False
        total_pairs += len(output_df)

    if chunk_number % 5 == 0:
        print(
            f"Processed chunks: {chunk_number} | "
            f"Pairs extracted: {total_pairs:,}"
        )


# ---------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET CREATION COMPLETE")
print("=" * 60)

print(f"AmazonHelp brand tweets : {total_brand:,}")
print(f"Customer-reply pairs    : {total_pairs:,}")
print(f"Output file             : {OUTPUT_FILE}")

if total_pairs > 0:

    df = pd.read_csv(OUTPUT_FILE)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nFirst 5 examples:")
    print(
        df[
            [
                "customer_text",
                "brand_text"
            ]
        ].head(5).to_string(index=False)
    )

print("\nStep 7 finished successfully.")
