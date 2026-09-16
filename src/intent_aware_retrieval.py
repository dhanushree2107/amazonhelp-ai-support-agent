import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CLEAN_FILE = "data/processed/amazonhelp_clean.csv"
SILVER_FILE = "data/processed/amazonhelp_silver.csv"


# ---------------------------------------------------------
# Intent-Aware Historical Retriever
# ---------------------------------------------------------

class IntentAwareRetriever:

    def __init__(self):

        print("=" * 70)
        print("STEP 14.2 - INTENT-AWARE HISTORICAL RETRIEVER")
        print("=" * 70)

        # -------------------------------------------------
        # Load clean conversations
        # -------------------------------------------------

        print("\nLoading historical conversations...")

        clean_df = pd.read_csv(
            CLEAN_FILE,
            dtype=str,
            keep_default_na=False
        )

        # -------------------------------------------------
        # Load silver intents
        # -------------------------------------------------

        print("Loading silver intent labels...")

        silver_df = pd.read_csv(
            SILVER_FILE,
            dtype=str,
            keep_default_na=False
        )

        # -------------------------------------------------
        # Keep only required columns
        # -------------------------------------------------

        silver_df = silver_df[
            [
                "customer_tweet_id",
                "silver_intent"
            ]
        ]

        # -------------------------------------------------
        # Merge intent with conversations
        # -------------------------------------------------

        self.df = clean_df.merge(
            silver_df,
            on="customer_tweet_id",
            how="left"
        )

        # Remove empty messages
        self.df = self.df[
            self.df["customer_text_clean"].str.strip() != ""
        ].copy()

        # Remove missing intent
        self.df = self.df[
            self.df["silver_intent"].str.strip() != ""
        ].copy()

        self.df = self.df.reset_index(
            drop=True
        )

        print(
            f"Usable conversations: {len(self.df):,}"
        )

        print("\nIntent distribution:")

        print(
            self.df["silver_intent"]
            .value_counts()
            .to_string()
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
    # Search using predicted intent
    # -----------------------------------------------------

    def search(
        self,
        query,
        intent,
        top_k=5
    ):

        # -------------------------------------------------
        # Filter historical examples by intent
        # -------------------------------------------------

        intent_df = self.df[
            self.df["silver_intent"] == intent
        ]

        if len(intent_df) == 0:

            print(
                f"No historical examples found "
                f"for intent: {intent}"
            )

            return pd.DataFrame()

        # -------------------------------------------------
        # Get original indices
        # -------------------------------------------------

        indices = intent_df.index

        intent_matrix = self.matrix[
            indices
        ]

        # -------------------------------------------------
        # Convert query to TF-IDF
        # -------------------------------------------------

        query_vector = self.vectorizer.transform(
            [query]
        )

        # -------------------------------------------------
        # Calculate similarity
        # -------------------------------------------------

        scores = cosine_similarity(
            query_vector,
            intent_matrix
        )[0]

        # -------------------------------------------------
        # Get top results
        # -------------------------------------------------

        top_positions = scores.argsort()[
            ::-1
        ][:top_k]

        top_indices = indices[
            top_positions
        ]

        results = self.df.loc[
            top_indices
        ].copy()

        results["similarity"] = scores[
            top_positions
        ]

        # -------------------------------------------------
        # Return useful columns
        # -------------------------------------------------

        return results[
            [
                "customer_tweet_id",
                "customer_text_clean",
                "brand_text",
                "silver_intent",
                "similarity"
            ]
        ]


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

def main():

    retriever = IntentAwareRetriever()

    test_cases = [

        (
            "Where is my order? It is late.",
            "delivery_issue"
        ),

        (
            "I want a refund for my order.",
            "return_refund"
        ),

        (
            "My Fire TV is not working.",
            "technical_issue"
        ),

        (
            "I cannot access my Amazon account.",
            "account_access"
        )

    ]

    # -----------------------------------------------------
    # Run test cases
    # -----------------------------------------------------

    for query, intent in test_cases:

        print("\n" + "=" * 70)

        print(
            f"QUERY: {query}"
        )

        print(
            f"PREDICTED INTENT: {intent}"
        )

        print("=" * 70)

        results = retriever.search(
            query,
            intent,
            top_k=3
        )

        for rank, (_, row) in enumerate(
            results.iterrows(),
            start=1
        ):

            print(
                f"\n--- Result {rank} ---"
            )

            print(
                f"Similarity: "
                f"{row['similarity']:.4f}"
            )

            print(
                f"Customer: "
                f"{row['customer_text_clean']}"
            )

            print(
                f"AmazonHelp: "
                f"{row['brand_text']}"
            )

            print(
                f"Intent: "
                f"{row['silver_intent']}"
            )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":

    main()
