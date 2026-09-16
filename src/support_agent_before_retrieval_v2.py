
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# IMPORTANT:
# Use the improved rule-based classifier from intent_rules.py
from intent_rules import rule_based_intent


# ============================================================
# WINDOWS UTF-8 OUTPUT
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CLEAN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "amazonhelp_clean.csv"
)

SILVER_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "amazonhelp_silver.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "tfidf_intent_model_v2.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading intent model...")

model = joblib.load(MODEL_FILE)

print("Intent model loaded.")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = (
        text
        .replace("&amp;", "&")
        .replace("\n", " ")
        .replace("\r", " ")
    )

    import re

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    # Remove Twitter mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD RETRIEVAL DATA
# ============================================================

print("Loading historical AmazonHelp conversations...")

clean_df = pd.read_csv(
    CLEAN_FILE,
    dtype=str,
    keep_default_na=False
)

silver_df = pd.read_csv(
    SILVER_FILE,
    dtype=str,
    keep_default_na=False
)

print(
    f"Historical conversations loaded: {len(clean_df):,}"
)


# ============================================================
# MERGE SILVER INTENTS
# ============================================================

if "silver_intent" in silver_df.columns:

    retrieval_df = clean_df.merge(
        silver_df[
            [
                "customer_tweet_id",
                "silver_intent"
            ]
        ],
        on="customer_tweet_id",
        how="left"
    )

else:

    retrieval_df = clean_df.copy()

    retrieval_df["silver_intent"] = "other"


# ============================================================
# CLEAN RETRIEVAL DATA
# ============================================================

retrieval_df["customer_text_clean"] = (
    retrieval_df["customer_text_clean"]
    .fillna("")
    .astype(str)
)

retrieval_df["customer_text"] = (
    retrieval_df["customer_text"]
    .fillna("")
    .astype(str)
)

retrieval_df["brand_text"] = (
    retrieval_df["brand_text"]
    .fillna("")
    .astype(str)
)

retrieval_df["silver_intent"] = (
    retrieval_df["silver_intent"]
    .fillna("other")
    .astype(str)
)

retrieval_df = retrieval_df[
    retrieval_df["customer_text_clean"].str.strip() != ""
].copy()

retrieval_df = retrieval_df.reset_index(drop=True)

print(
    f"Usable conversations: {len(retrieval_df):,}"
)


# ============================================================
# BUILD TF-IDF RETRIEVAL INDEX
# ============================================================

print("Building retrieval index...")

retrieval_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=150000
)

retrieval_matrix = retrieval_vectorizer.fit_transform(
    retrieval_df["customer_text_clean"]
)

print(
    "Retrieval index shape:",
    retrieval_matrix.shape
)


# ============================================================
# CLASSIFY MESSAGE
# ============================================================

def classify_message(text):

    cleaned = clean_text(text)

    # --------------------------------------------------------
    # FIRST: HIGH-CONFIDENCE RULE-BASED CLASSIFIER
    # --------------------------------------------------------

    rule_intent, rule_confidence = rule_based_intent(
        cleaned
    )

    if rule_intent is not None:

        return {
            "intent": rule_intent,
            "confidence": rule_confidence,
            "source": "RULE"
        }

    # --------------------------------------------------------
    # SECOND: MACHINE LEARNING CLASSIFIER
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        [cleaned]
    )[0]

    classes = model.classes_

    best_index = probabilities.argmax()

    predicted_intent = classes[best_index]

    confidence = float(
        probabilities[best_index]
    )

    return {
        "intent": predicted_intent,
        "confidence": confidence,
        "source": "ML"
    }


# ============================================================
# RETRIEVE SIMILAR HISTORICAL CONVERSATIONS
# ============================================================

def retrieve_similar(
    query,
    predicted_intent,
    top_k=3
):

    cleaned_query = clean_text(query)

    # --------------------------------------------------------
    # Prefer conversations with the same predicted intent
    # --------------------------------------------------------

    candidate_df = retrieval_df[
        retrieval_df["silver_intent"]
        == predicted_intent
    ].copy()

    # If there are too few candidates,
    # fall back to the complete dataset.
    if len(candidate_df) < top_k:

        candidate_df = retrieval_df.copy()

    candidate_df = candidate_df.reset_index(drop=True)

    # --------------------------------------------------------
    # Build candidate TF-IDF matrix
    # --------------------------------------------------------

    candidate_matrix = retrieval_vectorizer.transform(
        candidate_df["customer_text_clean"]
    )

    query_vector = retrieval_vectorizer.transform(
        [cleaned_query]
    )

    similarities = cosine_similarity(
        query_vector,
        candidate_matrix
    )[0]

    # --------------------------------------------------------
    # Optional retrieval boosts
    #
    # These do NOT change intent classification.
    # They only help choose better historical evidence.
    # --------------------------------------------------------

    query_lower = cleaned_query.lower()

    # --------------------------------------------------------
    # DELIVERY BOOST
    # --------------------------------------------------------

    delivery_phrases = [
        "where is my order",
        "where's my order",
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
        "where's my package",
        "where is my parcel",
        "where's my parcel",
        "tracking",
        "delivery",
        "delayed",
        "late"
    ]

    if predicted_intent == "delivery_issue":

        for i in range(len(candidate_df)):

            candidate_text = (
                candidate_df.iloc[i]["customer_text_clean"]
                .lower()
            )

            query_has_delivery = any(
                phrase in query_lower
                for phrase in delivery_phrases
            )

            candidate_has_delivery = any(
                phrase in candidate_text
                for phrase in delivery_phrases
            )

            if (
                query_has_delivery
                and candidate_has_delivery
            ):
                similarities[i] += 0.15

    # --------------------------------------------------------
    # ORDER PLACEMENT BOOST
    # --------------------------------------------------------

    order_phrases = [
        "place my order",
        "place an order",
        "place order",
        "not able to place",
        "unable to place",
        "cannot place",
        "can't place",
        "cant place",
        "how to order",
        "how do i order",
        "want to place an order"
    ]

    if predicted_intent == "order_issue":

        query_has_order_phrase = any(
            phrase in query_lower
            for phrase in order_phrases
        )

        if query_has_order_phrase:

            for i in range(len(candidate_df)):

                candidate_text = (
                    candidate_df.iloc[i]["customer_text_clean"]
                    .lower()
                )

                candidate_has_order_phrase = any(
                    phrase in candidate_text
                    for phrase in order_phrases
                )

                if candidate_has_order_phrase:
                    similarities[i] += 0.30

    # --------------------------------------------------------
    # GET TOP RESULTS
    # --------------------------------------------------------

    top_indices = similarities.argsort()[
        ::-1
    ][:top_k]

    results = []

    for index in top_indices:

        row = candidate_df.iloc[index]

        results.append(
            {
                "customer_text": row[
                    "customer_text"
                ],

                "brand_text": row[
                    "brand_text"
                ],

                "similarity": float(
                    similarities[index]
                ),

                "intent": row[
                    "silver_intent"
                ],

                "customer_tweet_id": row[
                    "customer_tweet_id"
                ]
            }
        )

    return results


# ============================================================
# ESCALATION POLICY
# ============================================================

def decide_escalation(
    query,
    classification,
    evidence
):

    intent = classification["intent"]

    confidence = classification["confidence"]

    # --------------------------------------------------------
    # SECURITY / ACCOUNT COMPROMISE
    # --------------------------------------------------------

    security_keywords = [

        "hacked",
        "hack",
        "stolen",
        "fraud",
        "fraudulent",
        "unauthorized",
        "unauthorised",
        "account compromised",
        "someone changed my email",
        "someone changed my password",
        "someone accessed my account",
        "someone has access",
        "identity theft"
    ]

    query_lower = clean_text(query).lower()

    for keyword in security_keywords:

        if keyword in query_lower:

            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason": (
                    "Security or account-compromise "
                    "issue requires human review."
                )
            }

    # --------------------------------------------------------
    # LOW CLASSIFICATION CONFIDENCE
    # --------------------------------------------------------

    if confidence < 0.55:

        return {
            "decision": "ESCALATE_TO_HUMAN",
            "reason": (
                "Intent classification confidence "
                "is below the safe threshold."
            )
        }

    # --------------------------------------------------------
    # NO HISTORICAL EVIDENCE
    # --------------------------------------------------------

    if not evidence:

        return {
            "decision": "ESCALATE_TO_HUMAN",
            "reason": (
                "No sufficiently similar historical "
                "support conversation was found."
            )
        }

    # --------------------------------------------------------
    # LOW RETRIEVAL SIMILARITY
    # --------------------------------------------------------

    top_similarity = evidence[0]["similarity"]

    if top_similarity < 0.20:

        return {
            "decision": "ESCALATE_TO_HUMAN",
            "reason": (
                "Historical evidence is too weak "
                "to safely draft an automated response."
            )
        }

    # --------------------------------------------------------
    # SENSITIVE INTENTS
    # --------------------------------------------------------

    sensitive_intents = {
        "return_refund",
        "payment_billing"
    }

    if intent in sensitive_intents:

        return {
            "decision": "ESCALATE_TO_HUMAN",
            "reason": (
                "Financial, refund, or payment issue "
                "requires human review."
            )
        }

    # --------------------------------------------------------
    # OTHERWISE AUTO HANDLE
    # --------------------------------------------------------

    return {
        "decision": "AUTO_HANDLE",
        "reason": (
            "High-confidence intent with sufficiently "
            "similar historical evidence."
        )
    }


# ============================================================
# HISTORICAL RESPONSE CLEANING
# ============================================================

def clean_historical_reply(reply):

    if pd.isna(reply):
        return ""

    reply = str(reply).strip()

    import re

    # Remove leading Twitter handle
    reply = re.sub(
        r"^@\w+\s*",
        "",
        reply
    )

    # Remove URLs
    reply = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        reply
    )

    # Normalize spaces
    reply = re.sub(
        r"\s+",
        " ",
        reply
    ).strip()

    return reply


# ============================================================
# GENERATE GROUNDED REPLY
# ============================================================

def generate_reply(
    query,
    classification,
    evidence
):

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not evidence:

        return (
            "I'm sorry you're experiencing this issue. "
            "I'll route this to our support team for "
            "further assistance."
        )

    # --------------------------------------------------------
    # Use strongest historical resolution
    # --------------------------------------------------------

    best_evidence = evidence[0]

    historical_reply = clean_historical_reply(
        best_evidence["brand_text"]
    )

    # --------------------------------------------------------
    # If historical response is usable,
    # ground the answer in it.
    # --------------------------------------------------------

    if historical_reply:

        return historical_reply

    # --------------------------------------------------------
    # Generic fallback
    # --------------------------------------------------------

    return (
        "I'm sorry you're experiencing this issue. "
        "Our support team can help you resolve it."
    )


# ============================================================
# RUN COMPLETE SUPPORT AGENT
# ============================================================

def run_agent(query):

    # --------------------------------------------------------
    # STEP 1: CLASSIFY
    # --------------------------------------------------------

    classification = classify_message(
        query
    )

    # --------------------------------------------------------
    # STEP 2: RETRIEVE HISTORICAL CASES
    # --------------------------------------------------------

    evidence = retrieve_similar(
        query,
        classification["intent"],
        top_k=3
    )

    # --------------------------------------------------------
    # STEP 3: ESCALATION DECISION
    # --------------------------------------------------------

    escalation = decide_escalation(
        query,
        classification,
        evidence
    )

    # --------------------------------------------------------
    # STEP 4: GENERATE GROUNDED REPLY
    # --------------------------------------------------------

    reply = generate_reply(
        query,
        classification,
        evidence
    )

    return {
        "query": query,

        "intent": classification[
            "intent"
        ],

        "confidence": classification[
            "confidence"
        ],

        "classification_source": classification[
            "source"
        ],

        "decision": escalation[
            "decision"
        ],

        "escalation_reason": escalation[
            "reason"
        ],

        "reply": reply,

        "evidence": evidence
    }


# ============================================================
# DISPLAY RESULT
# ============================================================

def print_result(result):

    print("\n" + "=" * 70)

    print(
        "CUSTOMER MESSAGE:"
    )

    print(
        result["query"]
    )

    print("\n" + "-" * 70)

    print(
        "PREDICTED INTENT:"
    )

    print(
        result["intent"]
    )

    print(
        f"CONFIDENCE: "
        f"{result['confidence']:.4f}"
    )

    print(
        "CLASSIFICATION SOURCE:"
    )

    print(
        result["classification_source"]
    )

    print("\n" + "-" * 70)

    print(
        "DECISION:"
    )

    print(
        result["decision"]
    )

    print(
        "REASON:"
    )

    print(
        result["escalation_reason"]
    )

    print("\n" + "-" * 70)

    print(
        "DRAFT REPLY:"
    )

    print(
        result["reply"]
    )

    print("\n" + "-" * 70)

    print(
        "HISTORICAL EVIDENCE:"
    )

    for i, item in enumerate(
        result["evidence"],
        start=1
    ):

        print(
            f"\nEvidence {i}"
        )

        print(
            f"Similarity: "
            f"{item['similarity']:.4f}"
        )

        print(
            f"Intent: "
            f"{item['intent']}"
        )

        print(
            "Customer:"
        )

        print(
            item["customer_text"]
        )

        print(
            "AmazonHelp:"
        )

        print(
            item["brand_text"]
        )

    print(
        "=" * 70
    )


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    test_queries = [

        "Where is my order? It was supposed to arrive yesterday.",

        "My Fire TV is not working.",

        "I want a refund for my order.",

        "Someone hacked my Amazon account and changed my email.",

        "Why am I not able to place my order?"
    ]

    print("\n")
    print("=" * 70)
    print("AMAZONHELP AI SUPPORT AGENT")
    print("=" * 70)

    for query in test_queries:

        result = run_agent(
            query
        )

        print_result(
            result
        )

