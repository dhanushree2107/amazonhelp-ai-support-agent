import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


SILVER_FILE = "data/processed/amazonhelp_silver.csv"
MODEL_FILE = "data/processed/tfidf_intent_model_v2.joblib"

RANDOM_STATE = 42
MAX_PER_CLASS = 5000


def main():

    print("=" * 70)
    print("STEP 16 - IMPROVED INTENT CLASSIFIER")
    print("=" * 70)

    print("\nLoading silver dataset...")

    df = pd.read_csv(
        SILVER_FILE,
        dtype=str,
        keep_default_na=False
    )

    df["customer_text_clean"] = df["customer_text_clean"].fillna("").astype(str)
    df["silver_intent"] = df["silver_intent"].fillna("").astype(str)

    df = df[
        (df["customer_text_clean"].str.strip() != "") &
        (df["silver_intent"].str.strip() != "")
    ].copy()

    print(f"Total usable rows: {len(df):,}")

    # ---------------------------------------------------------
    # Improve obvious delivery/order confusion
    # ---------------------------------------------------------

    delivery_patterns = [
        "where is my order",
        "order is late",
        "order was late",
        "order is delayed",
        "order was delayed",
        "hasn't arrived",
        "hasnt arrived",
        "not arrived",
        "haven't received",
        "havent received",
        "didn't receive",
        "didnt receive",
        "where is my package",
        "where is my parcel",
        "tracking",
        "delivery",
        "delivered but",
        "supposed to arrive",
        "supposed to be delivered"
    ]

    mask = (
        df["customer_text_clean"]
        .str.lower()
        .apply(
            lambda text: any(
                pattern in text
                for pattern in delivery_patterns
            )
        )
    )

    affected = mask.sum()

    df.loc[mask, "silver_intent"] = "delivery_issue"

    print(
        f"\nReassigned obvious delivery messages: {affected:,}"
    )

    # ---------------------------------------------------------
    # Remove extremely dominant "other" class from training
    # ---------------------------------------------------------

    print("\nOriginal class distribution:")

    print(
        df["silver_intent"]
        .value_counts()
        .sort_values(ascending=False)
    )

    print("\nBalancing classes...")

    balanced_parts = []

    for intent, group in df.groupby("silver_intent"):

        if len(group) > MAX_PER_CLASS:
            group = group.sample(
                n=MAX_PER_CLASS,
                random_state=RANDOM_STATE
            )

        balanced_parts.append(group)

    balanced_df = pd.concat(
        balanced_parts,
        ignore_index=True
    )

    print("\nBalanced distribution:")

    print(
        balanced_df["silver_intent"]
        .value_counts()
        .sort_values(ascending=False)
    )

    # ---------------------------------------------------------
    # Train / test split
    # ---------------------------------------------------------

    X = balanced_df["customer_text_clean"]
    y = balanced_df["silver_intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")

    # ---------------------------------------------------------
    # TF-IDF + Logistic Regression
    # ---------------------------------------------------------

    print("\nTraining improved TF-IDF classifier...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=200000,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ])

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    print("\nEvaluating...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            digits=4
        )
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(f"\n{MODEL_FILE}")


if __name__ == "__main__":
    main()
