import pandas as pd

DATA_PATH = "data/twcs.csv"

df = pd.read_csv(DATA_PATH, nrows=10000)

print("Dataset loaded successfully!")
print("Rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nInbound distribution:")
print(df["inbound"].value_counts())