import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_FILE = "data/processed/amazonhelp_clean.csv"


# ---------------------------------------------------------
# Historical Retriever
# ---------------------------------------------------------

class HistoricalRetriever:

    def __init__(self):

        print("=" * 70)
        print("STEP 14.1 - HISTORICAL REPLY RETRIEVER")
        print("=" * 70)

        # -------------------------------------------------
        # Load historical conversations
        # -------------------------------------------------

        print("\nLoading AmazonHelp conversations...")

        self.df = pd.read_csv(
            DATA_FILE,
            dtype=str,
            keep_default_na=False
        )

        # Remove empty customer messages
        self.df = self.df[
            self.df["customer_text_clean"].str.strip() != ""
        ].copy()

        # Reset index
        self.df = self.df.reset_index(drop=True)

        print(
            f"Historical conversations: {len(self.df):,}"
        )

        # -------------------------------------------------
        # Build TF-IDF index
        # -------------------------------------------------

        print("\nBuilding TF-IDF index...")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=150000,
            sublinear_tf=True
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["customer_text_clean"]
        )

        print("TF-IDF index created.")

        print(
            f"Matrix shape: {self.matrix.shape}"
        )

    # -----------------------------------------------------
    # Search similar historical conversations
    # -----------------------------------------------------

    def search(
        self,
        query,
        top_k=5
    ):

        # Convert new customer message to TF-IDF
        query_vector = self.vectorizer.transform(
            [query]
        )

        # Calculate similarity
        scores = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        # Get top matching conversations
        top_indices = scores.argsort()[
            ::-1
        ][:top_k]

        # Select rows
        results = self.df.iloc[
            top_indices
        ].copy()

        # Add similarity score
        results["similarity"] = scores[
            top_indices
        ]

        # -------------------------------------------------
        # Select output columns
        # -------------------------------------------------

        columns = [
            "customer_tweet_id",
            "customer_text_clean",
            "brand_text"
        ]

        # Add silver intent if available
        if "silver_intent" in results.columns:
            columns.append("silver_intent")

        # IMPORTANT:
        # Keep similarity column
        columns.append("similarity")

        return results[
            columns
        ]


# ---------------------------------------------------------
# Test the retriever
# ---------------------------------------------------------

def main():

    retriever = HistoricalRetriever()

    # Example customer queries
    test_queries = [

        "Where is my order? It is late.",

        "I want a refund for my order.",

        "My Fire TV is not working.",

        "I cannot access my Amazon account."

    ]

    # -----------------------------------------------------
    # Run searches
    # -----------------------------------------------------

    for query in test_queries:

        print("\n" + "=" * 70)

        print(
            f"QUERY: {query}"
        )

        print("=" * 70)

        results = retriever.search(
            query,
            top_k=3
        )

        # Display results
        for rank, (_, row) in enumerate(
            results.iterrows(),
            start=1
        ):

            print(
                f"\n--- Result {rank} ---"
            )

            print(
                f"Similarity: {row['similarity']:.4f}"
            )

            print(
                f"Customer: "
                f"{row['customer_text_clean']}"
            )

            print(
                f"AmazonHelp: "
                f"{row['brand_text']}"
            )

            if "silver_intent" in results.columns:

                print(
                    f"Intent: "
                    f"{row['silver_intent']}"
                )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":

    main()
