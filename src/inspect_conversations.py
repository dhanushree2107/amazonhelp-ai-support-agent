import pandas as pd

DATA_PATH = "data/twcs.csv"

df = pd.read_csv(
    DATA_PATH,
    nrows=100000,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
)

brand = "AmazonHelp"

brand_df = df[df["author_id"] == brand]

print("=" * 70)
print("AMAZONHELP SAMPLE TWEETS")
print("=" * 70)

print("\nNumber of AmazonHelp tweets:", len(brand_df))

print("\nFirst 10 AmazonHelp tweets:\n")

for _, row in brand_df.head(10).iterrows():

    print("-" * 70)

    print("Tweet ID:", row["tweet_id"])
    print("Author:", row["author_id"])
    print("Inbound:", row["inbound"])
    print("Created:", row["created_at"])

    print("Text:")
    print(row["text"])

    print("Response Tweet ID:", row["response_tweet_id"])
    print("In Response To:", row["in_response_to_tweet_id"])
