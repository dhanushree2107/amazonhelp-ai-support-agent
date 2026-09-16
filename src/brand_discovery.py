import pandas as pd

DATA_PATH = "data/twcs.csv"

df = pd.read_csv(DATA_PATH, nrows=100000)

print("Rows loaded:", len(df))

print("\nUnique authors:")
print(df["author_id"].nunique())

print("\nInbound tweets:")
print(df[df["inbound"] == True]["author_id"].nunique())

print("\nOutbound tweets:")
print(df[df["inbound"] == False]["author_id"].nunique())

print("\nTop authors by tweet count:")
print(df["author_id"].value_counts().head(30))