import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


GOLDEN_FILE = "data/golden/amazonhelp_golden.csv"


def main():
    print("=" * 70)
    print("STEP 13.1 - TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    # Load golden labels
    df = pd.read_csv(
        GOLDEN_FILE,
        dtype=str,
        keep_default_na=False
    )

    # Keep only manually reviewed examples
    df = df[df["gold_intent"].str.strip() != ""]

    print(f"\nLabeled examples: {len(df)}")

    print("\nIntent distribution:")
    print(df["gold_intent"].value_counts())

    X = df["customer_text_clean"]
    y = df["gold_intent"]

    # Because we currently have very few examples per class,
    # use a simple train/test split only when possible.
    if len(df) >= 30 and y.value_counts().min() >= 2:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

        model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000
                )
            )
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        print("\nAccuracy:")
        print(accuracy_score(y_test, predictions))

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

    else:
        print("\nNot enough labeled data for a reliable train/test evaluation.")
        print("The baseline pipeline is ready, but the golden set needs")
        print("more reviewed examples before reporting metrics.")


if __name__ == "__main__":
    main()
