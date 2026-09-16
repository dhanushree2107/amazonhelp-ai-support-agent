import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


INPUT_FILE = "data/processed/amazonhelp_silver.csv"
MODEL_FILE = "data/processed/tfidf_intent_model.joblib"

SAMPLES_PER_INTENT = 3000


def main():

    print("=" * 70)
    print("STEP 13.4 - TRAIN AND SAVE INTENT MODEL")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    print(f"\nTotal silver rows: {len(df):,}")

    # Remove empty messages
    df = df[
        df["customer_text_clean"].str.strip() != ""
    ].copy()

    # Balanced sampling
    samples = []

    for intent in df["silver_intent"].unique():

        intent_df = df[
            df["silver_intent"] == intent
        ]

        n = min(
            len(intent_df),
            SAMPLES_PER_INTENT
        )

        samples.append(
            intent_df.sample(
                n=n,
                random_state=42
            )
        )

    df = pd.concat(
        samples,
        ignore_index=True
    )

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print(
        f"Training rows: {len(df):,}"
    )

    print("\nClass distribution:")
    print(
        df["silver_intent"]
        .value_counts()
        .to_string()
    )

    X = df["customer_text_clean"]
    y = df["silver_intent"]

    # Model
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=100000,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    print("\nTraining final baseline model...")

    model.fit(X, y)

    print("Training complete.")

    # Save
    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"\nModel saved to: {MODEL_FILE}"
    )

    print("\nModel classes:")

    classifier = model.named_steps["classifier"]

    for label in classifier.classes_:
        print(f"  - {label}")

    print("\n" + "=" * 70)
    print("STEP 13.4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
