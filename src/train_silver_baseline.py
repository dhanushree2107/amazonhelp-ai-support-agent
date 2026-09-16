import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


INPUT_FILE = "data/processed/amazonhelp_silver.csv"

SAMPLES_PER_INTENT = 3000


def main():

    print("=" * 70)
    print("STEP 13.3 - TF-IDF + LOGISTIC REGRESSION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load silver dataset
    # ---------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    print(f"\nTotal silver rows: {len(df):,}")

    # ---------------------------------------------------------
    # 2. Remove empty messages
    # ---------------------------------------------------------

    df = df[
        df["customer_text_clean"].str.strip() != ""
    ].copy()

    # ---------------------------------------------------------
    # 3. Create balanced subset
    # ---------------------------------------------------------

    samples = []

    for intent in df["silver_intent"].unique():

        intent_df = df[
            df["silver_intent"] == intent
        ]

        n = min(
            len(intent_df),
            SAMPLES_PER_INTENT
        )

        sampled = intent_df.sample(
            n=n,
            random_state=42
        )

        samples.append(sampled)

    df = pd.concat(
        samples,
        ignore_index=True
    )

    # Shuffle
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. Show distribution
    # ---------------------------------------------------------

    print(f"\nTraining/evaluation rows: {len(df):,}")

    print("\nClass distribution:")
    print(
        df["silver_intent"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # 5. Prepare X and y
    # ---------------------------------------------------------

    X = df["customer_text_clean"]
    y = df["silver_intent"]

    # ---------------------------------------------------------
    # 6. Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain rows: {len(X_train):,}")
    print(f"Test rows:  {len(X_test):,}")

    # ---------------------------------------------------------
    # 7. TF-IDF + Logistic Regression
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 8. Train
    # ---------------------------------------------------------

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # ---------------------------------------------------------
    # 9. Predict
    # ---------------------------------------------------------

    predictions = model.predict(X_test)

    # ---------------------------------------------------------
    # 10. Accuracy
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    # ---------------------------------------------------------
    # 11. Classification report
    # ---------------------------------------------------------

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # 12. Confusion matrix
    # ---------------------------------------------------------

    print("\nConfusion matrix:")

    labels = sorted(
        y.unique()
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print(cm_df)


if __name__ == "__main__":
    main()
