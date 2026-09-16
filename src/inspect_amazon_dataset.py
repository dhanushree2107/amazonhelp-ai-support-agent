import pandas as pd

INPUT_FILE = "data/processed/amazonhelp_pairs.csv"

print("=" * 60)
print("STEP 8 - INSPECT AMAZONHELP DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"\nTotal rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

print("\nColumns:")
for col in df.columns:
    print(f"- {col}")

# ---------------------------------------------------------
# BASIC DATA QUALITY
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("DATA QUALITY")
print("-" * 60)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate customer tweets:")
print(df["customer_tweet_id"].duplicated().sum())

print("\nDuplicate customer texts:")
print(df["customer_text"].duplicated().sum())

# ---------------------------------------------------------
# TEXT STATISTICS
# ---------------------------------------------------------

df["customer_text"] = df["customer_text"].fillna("").astype(str)
df["brand_text"] = df["brand_text"].fillna("").astype(str)

df["customer_length"] = df["customer_text"].str.len()
df["brand_length"] = df["brand_text"].str.len()

print("\n" + "-" * 60)
print("TEXT STATISTICS")
print("-" * 60)

print(
    f"Customer average length: "
    f"{df['customer_length'].mean():.1f}"
)

print(
    f"Customer shortest: "
    f"{df['customer_length'].min()}"
)

print(
    f"Customer longest: "
    f"{df['customer_length'].max()}"
)

print(
    f"Brand average length: "
    f"{df['brand_length'].mean():.1f}"
)

# ---------------------------------------------------------
# SAMPLE CUSTOMER MESSAGES
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("CUSTOMER MESSAGE SAMPLES")
print("-" * 60)

sample = df.sample(
    n=min(30, len(df)),
    random_state=42
)

for i, text in enumerate(sample["customer_text"], start=1):
    print(f"\n{i}. {text}")

# ---------------------------------------------------------
# SAMPLE REPLY PAIRS
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("CUSTOMER -> AMAZONHELP REPLY PAIRS")
print("-" * 60)

for i, row in df.sample(
    n=min(10, len(df)),
    random_state=10
).iterrows():

    print(f"\nPAIR {i}")
    print(f"Customer : {row['customer_text']}")
    print(f"AmazonHelp: {row['brand_text']}")

# ---------------------------------------------------------
# MOST COMMON CUSTOMER WORDS
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("COMMON WORDS")
print("-" * 60)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    stop_words="english",
    max_features=30
)

X = vectorizer.fit_transform(df["customer_text"])

word_counts = X.sum(axis=0).A1

words = vectorizer.get_feature_names_out()

word_frequency = sorted(
    zip(words, word_counts),
    key=lambda x: x[1],
    reverse=True
)

for word, count in word_frequency:
    print(f"{word:20} {count}")

print("\n" + "=" * 60)
print("STEP 8 COMPLETE")
print("=" * 60)
