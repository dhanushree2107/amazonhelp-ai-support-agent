
import os
import sys
import re
from pathlib import Path

import joblib
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from intent_rules import rule_based_intent

LAST_REPLY_SOURCE = "unknown"


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
# GROQ API
# ============================================================

ENV_FILE = BASE_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_FILE
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)
GROQ_MODEL = "openai/gpt-oss-120b"

if GROQ_API_KEY:
    groq_client = Groq(
        api_key=GROQ_API_KEY,
        
    )
    print("Groq AI client ready.")
else:
    groq_client = None
    print(
        "WARNING: GROQ_API_KEY not found. "
        "AI replies will use fallback mode."
    )


# ============================================================
# LOAD INTENT CLASSIFIER
# ============================================================

print("Loading intent classifier...")

model = joblib.load(MODEL_FILE)

print("Loaded model type:", type(model))

print("Intent classifier loaded.")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.replace("&amp;", "&")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

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

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# CLEAN HISTORICAL REPLY
# ============================================================

def clean_historical_reply(reply):

    if pd.isna(reply):
        return ""

    reply = str(reply).strip()

    # Remove HTML entities
    reply = reply.replace("&amp;", "&")
    reply = reply.replace("&lt;", "<")
    reply = reply.replace("&gt;", ">")

    # Remove leading Twitter handles
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

    # Remove short agent signatures at end
    # Examples: ^SJ, ^GD, ^RR
    reply = re.sub(
        r"\s*\^[A-Za-z0-9]{1,5}\s*$",
        "",
        reply
    )

    # Remove extra spaces
    reply = re.sub(
        r"\s+",
        " ",
        reply
    ).strip()

    return reply


# ============================================================
# LOAD AMAZONHELP DATA
# ============================================================

print("Loading AmazonHelp data...")

clean_df = pd.read_csv(
    CLEAN_FILE,
    dtype=str,
    keep_default_na=False
)

print(
    f"Clean conversations loaded: "
    f"{len(clean_df):,}"
)


# ============================================================
# LOAD SILVER LABELS
# ============================================================

print("Loading silver labels...")

silver_df = pd.read_csv(
    SILVER_FILE,
    dtype=str,
    keep_default_na=False
)

print(
    f"Silver rows loaded: "
    f"{len(silver_df):,}"
)


# ============================================================
# MERGE INTENT LABELS
# ============================================================

print("Merging intent labels...")

if (
    "customer_tweet_id" in silver_df.columns
    and "silver_intent" in silver_df.columns
):

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
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "customer_text_clean",
    "customer_text",
    "brand_text",
    "silver_intent",
    "customer_tweet_id"
]

for column in required_columns:

    if column not in retrieval_df.columns:

        if column == "brand_text":
            retrieval_df[column] = ""

        elif column == "customer_tweet_id":
            retrieval_df[column] = ""

        elif column == "silver_intent":
            retrieval_df[column] = "other"

        else:
            raise ValueError(
                f"Required column missing: {column}"
            )


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
    retrieval_df["customer_text_clean"]
    .str.strip() != ""
].copy()

retrieval_df = retrieval_df.reset_index(
    drop=True
)

print(
    f"Usable conversation rows: "
    f"{len(retrieval_df):,}"
)


# ============================================================
# BUILD TF-IDF RETRIEVAL INDEX
# ============================================================

print("Building TF-IDF retrieval index...")

retrieval_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=150000
)

retrieval_matrix = (
    retrieval_vectorizer.fit_transform(
        retrieval_df["customer_text_clean"]
    )
)

print(
    "Retrieval matrix:",
    retrieval_matrix.shape
)


# ============================================================
# CLASSIFY MESSAGE
# ============================================================

def classify_message(text):

    cleaned = clean_text(text)

    # --------------------------------------------------------
    # RULE-BASED CLASSIFICATION FIRST
    # --------------------------------------------------------

    rule_intent, rule_confidence = (
        rule_based_intent(cleaned)
    )

    if rule_intent is not None:

        return {
            "intent": rule_intent,
            "confidence": float(rule_confidence),
            "source": "RULE"
        }

    # --------------------------------------------------------
    # MACHINE LEARNING CLASSIFICATION
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        [cleaned]
    )[0]

    classes = model.classes_

    best_index = probabilities.argmax()

    predicted_intent = classes[
        best_index
    ]

    confidence = float(
        probabilities[best_index]
    )

    return {
        "intent": predicted_intent,
        "confidence": confidence,
        "source": "ML"
    }


# ============================================================
# RETRIEVE SIMILAR CONVERSATIONS
# ============================================================

def retrieve_similar(
    query,
    predicted_intent,
    top_k=3
):

    cleaned_query = clean_text(query)

    # --------------------------------------------------------
    # Prefer same intent
    # --------------------------------------------------------

    candidate_df = retrieval_df[
        retrieval_df["silver_intent"]
        == predicted_intent
    ].copy()

    # Fallback to all conversations
    if len(candidate_df) < top_k:

        candidate_df = retrieval_df.copy()

    candidate_df = candidate_df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # TF-IDF VECTORS
    # --------------------------------------------------------

    candidate_matrix = (
        retrieval_vectorizer.transform(
            candidate_df[
                "customer_text_clean"
            ]
        )
    )

    query_vector = (
        retrieval_vectorizer.transform(
            [cleaned_query]
        )
    )

    similarities = cosine_similarity(
        query_vector,
        candidate_matrix
    )[0]

    # Ranking score starts with similarity
    ranking_scores = similarities.copy()

    query_lower = cleaned_query.lower()


    # ========================================================
    # DELIVERY BOOST
    # ========================================================

    delivery_phrases = [

        "where is my order",
        "where's my order",
        "where is my package",
        "where's my package",
        "where is my parcel",
        "where's my parcel",

        "order is late",
        "order was late",

        "order is delayed",
        "order was delayed",

        "hasn't arrived",
        "hasnt arrived",

        "haven't received",
        "havent received",

        "didn't receive",
        "didnt receive",

        "not arrived",

        "tracking",

        "delivery",

        "delayed",

        "late"
    ]

    if predicted_intent == "delivery_issue":

        query_has_delivery = any(
            phrase in query_lower
            for phrase in delivery_phrases
        )

        if query_has_delivery:

            for i in range(
                len(candidate_df)
            ):

                candidate_text = (
                    candidate_df.iloc[i][
                        "customer_text_clean"
                    ]
                    .lower()
                )

                candidate_has_delivery = any(
                    phrase in candidate_text
                    for phrase in delivery_phrases
                )

                if candidate_has_delivery:

                    ranking_scores[i] += 0.10


    # ========================================================
    # ORDER PLACEMENT BOOST
    # ========================================================

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

        query_has_order = any(
            phrase in query_lower
            for phrase in order_phrases
        )

        if query_has_order:

            for i in range(
                len(candidate_df)
            ):

                candidate_text = (
                    candidate_df.iloc[i][
                        "customer_text_clean"
                    ]
                    .lower()
                )

                candidate_has_order = any(
                    phrase in candidate_text
                    for phrase in order_phrases
                )

                if candidate_has_order:

                    ranking_scores[i] += 0.15


    # ========================================================
    # TECHNICAL DEVICE BOOST
    # ========================================================

    technical_terms = [

        "not working",

        "fire tv",

        "fire stick",

        "alexa",

        "echo",

        "kindle",

        "device",

        "screen",

        "error",

        "app not working",

        "cannot connect"
    ]

    if predicted_intent == "technical_issue":

        query_terms = [
            term
            for term in technical_terms
            if term in query_lower
        ]

        if query_terms:

            for i in range(
                len(candidate_df)
            ):

                candidate_text = (
                    candidate_df.iloc[i][
                        "customer_text_clean"
                    ]
                    .lower()
                )

                matching_terms = sum(
                    term in candidate_text
                    for term in query_terms
                )

                ranking_scores[i] += (
                    matching_terms * 0.05
                )


    # ========================================================
    # ACCOUNT SECURITY BOOST
    # ========================================================

    security_terms = [

        "hacked",

        "hack",

        "password",

        "email",

        "cannot login",

        "can't login",

        "cant login",

        "account",

        "unauthorized",

        "unauthorised"
    ]

    if predicted_intent == "account_access":

        query_terms = [
            term
            for term in security_terms
            if term in query_lower
        ]

        if query_terms:

            for i in range(
                len(candidate_df)
            ):

                candidate_text = (
                    candidate_df.iloc[i][
                        "customer_text_clean"
                    ]
                    .lower()
                )

                matching_terms = sum(
                    term in candidate_text
                    for term in query_terms
                )

                ranking_scores[i] += (
                    matching_terms * 0.04
                )


    # ========================================================
    # REFUND BOOST
    # ========================================================

    refund_terms = [

        "refund",

        "money back",

        "return",

        "returned",

        "refund status"
    ]

    if predicted_intent == "return_refund":

        query_terms = [
            term
            for term in refund_terms
            if term in query_lower
        ]

        if query_terms:

            for i in range(
                len(candidate_df)
            ):

                candidate_text = (
                    candidate_df.iloc[i][
                        "customer_text_clean"
                    ]
                    .lower()
                )

                matching_terms = sum(
                    term in candidate_text
                    for term in query_terms
                )

                ranking_scores[i] += (
                    matching_terms * 0.05
                )


    # ========================================================
    # GET TOP RESULTS
    # ========================================================

    top_indices = ranking_scores.argsort()[
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

                "ranking_score": float(
                    ranking_scores[index]
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

    confidence = classification[
        "confidence"
    ]

    query_lower = clean_text(
        query
    ).lower()


    # --------------------------------------------------------
    # SECURITY
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

        "identity theft"
    ]

    if any(
        keyword in query_lower
        for keyword in security_keywords
    ):

        return {
            "decision": "human",
            "reason": (
                "Security or account-compromise risk."
            )
        }


    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------

    if confidence < 0.55:

        return {
            "decision": "human",
            "reason": (
                "Low intent classification confidence."
            )
        }


    # --------------------------------------------------------
    # NO EVIDENCE
    # --------------------------------------------------------

    if not evidence:

        return {
            "decision": "human",
            "reason": (
                "No historical evidence found."
            )
        }


    # --------------------------------------------------------
    # LOW SIMILARITY
    # --------------------------------------------------------

    top_similarity = evidence[0][
        "similarity"
    ]

    if top_similarity < 0.20:

        return {
            "decision": "human",
            "reason": (
                "Historical evidence is too weak."
            )
        }


    # --------------------------------------------------------
    # SENSITIVE FINANCIAL INTENTS
    # --------------------------------------------------------

    sensitive_intents = {

        "return_refund",

        "payment_billing"
    }

    if intent in sensitive_intents:

        return {
            "decision": "human",
            "reason": (
                "Sensitive financial or refund issue."
            )
        }


    # --------------------------------------------------------
    # AUTO HANDLE
    # --------------------------------------------------------

    return {
        "decision": "auto",
        "reason": (
            "High-confidence intent with sufficient "
            "historical evidence."
        )
    }


# ============================================================
# BUILD EVIDENCE FOR AI
# ============================================================

def build_evidence_text(
    evidence,
    max_items=3
):

    if not evidence:
        return "No historical evidence available."

    evidence_parts = []

    for i, item in enumerate(
        evidence[:max_items],
        start=1
    ):

        customer = clean_text(
            item["customer_text"]
        )

        reply = clean_historical_reply(
            item["brand_text"]
        )

        if not reply:
            continue

        evidence_parts.append(
            f"""
Example {i}

Customer:
{customer}

Historical support response:
{reply}
""".strip()
        )

    if not evidence_parts:

        return "No usable historical responses available."

    return "\n\n".join(
        evidence_parts
    )


# ============================================================
# FALLBACK REPLIES
# ============================================================

def fallback_reply(
    query,
    classification,
    evidence
):

    intent = classification["intent"]

    if intent == "delivery_issue":
        return (
            "I'm sorry your order hasn't arrived as expected. "
            "Please check the latest order tracking details, "
            "and contact support if you need further assistance."
        )

    if intent == "technical_issue":
        return (
            "I'm sorry you're experiencing this issue. "
            "Could you share more details about what is happening, "
            "including any error message or troubleshooting steps "
            "you've already tried?"
        )

    if intent == "return_refund":
        return (
            "I'm sorry for the trouble. For refund or return "
            "assistance, please contact customer support so the "
            "order can be reviewed securely."
        )

    if intent == "payment_billing":
        return (
            "I'm sorry you're having a billing or payment issue. "
            "Please contact customer support so the account and "
            "transaction can be reviewed securely."
        )

    if intent == "account_access":
        return (
            "I'm sorry you're having trouble accessing your account. "
            "For security, please contact customer support directly "
            "for assistance."
        )

    if intent == "order_issue":
        return (
            "Sorry you're having trouble placing your order. "
            "Please share the device or platform you're using and "
            "any error message you see so we can help identify the issue."
        )

    if intent == "wrong_damaged_item":
        return (
            "I'm sorry there is an issue with the item you received. "
            "Please contact customer support so the order details and "
            "available options can be reviewed."
        )

    if intent == "cancellation_change":
        return (
            "I understand you'd like to change or cancel your order. "
            "Please check your order details or contact customer support "
            "for available options."
        )

    if intent == "prime_membership":
        return (
            "I'm sorry you're having an issue with your Prime membership. "
            "Please share a little more about the membership problem so we can help."
        )

    if intent == "customer_service":
        return (
            "I'm sorry about your experience. "
            "Please share more details about what happened so we can "
            "help review the situation."
        )

    return (
        "I'm sorry you're experiencing this issue. "
        "Please share a little more detail so we can help you further."
    )


# ============================================================
# GENERATE AI REPLY WITH GROQ
# ============================================================

def generate_ai_reply(
    query,
    classification,
    escalation,
    evidence
):

    global LAST_REPLY_SOURCE
    LAST_REPLY_SOURCE = "unknown"

    if groq_client is None:
        LAST_REPLY_SOURCE = "fallback_no_client"
        return fallback_reply(query, classification, evidence)

    intent = classification["intent"]
    decision = escalation["decision"]

    evidence_text = build_evidence_text(
        evidence,
        max_items=3
    )

    system_prompt = """
You are a customer support reply assistant.

Your task is to write ONE short, professional, helpful customer support response.

IMPORTANT RULES:

1. Use ONLY the customer message and the historical evidence provided.
2. Do not invent order details.
3. Do not invent links, phone numbers, email addresses, policies,
   refund amounts, delivery dates, or account information.
4. Do not mention Twitter, tweets, datasets, AI, Groq, retrieval,
   evidence, similarity scores, or intent classification.
5. Do not copy irrelevant information from historical examples.
6. Do not include agent signatures such as ^SJ or ^GD.
7. Do not include URLs unless the customer explicitly provided one
   and it is necessary to refer to it.
8. Keep the answer concise: maximum 2 or 3 sentences.
9. Be polite, natural and specific to the customer's problem.
10. Do not claim that an action has already been taken unless
    the historical evidence clearly supports that claim.
11. Use normal spaces between every word.
12. Return ONLY the final customer support response.
13. Do not start the response with labels such as "Reply:" or "Response:".
"""

    if decision == "human":
        escalation_instruction = """
This issue requires human support review.

Write a helpful acknowledgement and direct the customer to contact
customer support securely.

Do not claim that the issue is resolved.
Do not request sensitive information such as passwords or full card details.
"""
    else:
        escalation_instruction = """
This issue may be handled automatically.

Provide practical next-step guidance based on the strongest historical
support examples and the customer's exact problem.
"""

    user_prompt = f"""
CUSTOMER MESSAGE:
{clean_text(query)}

PREDICTED ISSUE TYPE:
{intent}

SUPPORT HANDLING:
{decision}

HANDLING INSTRUCTION:
{escalation_instruction}

HISTORICAL SUPPORT EVIDENCE:
{evidence_text}

Write the final customer support reply only.
"""

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.3,
            max_completion_tokens=150
        )

        reply = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        reply = re.sub(
            r"https?://\S+|www\.\S+",
            "",
            reply
        )

        reply = re.sub(
            r"\s*\^[A-Za-z0-9]{1,5}\s*$",
            "",
            reply
        )

        reply = re.sub(
            r"\s+",
            " ",
            reply
        ).strip()

        # Fix common word-joining artifacts.
        replacements = {
            "contactcustomer": "contact customer",
            "contactour": "contact our",
            "thecontact": "the contact",
            "tohelp": "to help",
            "wecan": "we can",
            "youcan": "you can",
            "pleasecheck": "please check",
            "youraccount": "your account",
            "yourorder": "your order",
            "historicalevidence": "historical evidence",
            "supportteam": "support team",
            "customerteam": "customer team",
            "contactsupport": "contact support"
        }

        for wrong, correct in replacements.items():
            reply = reply.replace(wrong, correct)

        reply = re.sub(
            r"\s+",
            " ",
            reply
        ).strip()

        # A response shorter than this is treated as likely truncated.
        if len(reply) < 60:
            LAST_REPLY_SOURCE = "fallback_short_response"
            return fallback_reply(
                query,
                classification,
                evidence
            )

        LAST_REPLY_SOURCE = "llm_generated"
        return reply

    except Exception as error:
        print("\nGroq generation error:", error)
        print("Using fallback reply...")

        LAST_REPLY_SOURCE = "fallback_api_error"

        return fallback_reply(
            query,
            classification,
            evidence
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
    # STEP 2: RETRIEVE
    # --------------------------------------------------------

    evidence = retrieve_similar(
        query=query,
        predicted_intent=classification[
            "intent"
        ],
        top_k=3
    )


    # --------------------------------------------------------
    # STEP 3: DECIDE ESCALATION
    # --------------------------------------------------------

    escalation = decide_escalation(
        query=query,
        classification=classification,
        evidence=evidence
    )


    # --------------------------------------------------------
    # STEP 4: GENERATE AI REPLY
    # --------------------------------------------------------

    reply = generate_ai_reply(
        query=query,
        classification=classification,
        escalation=escalation,
        evidence=evidence
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

        "reply_source": LAST_REPLY_SOURCE,

        "evidence": evidence
    }


# ============================================================
# DISPLAY RESULT
# ============================================================

def print_result(result):

    print("\n")

    print("=" * 70)

    print("CUSTOMER:")

    print(
        result["query"]
    )


    print("\nINTENT:")

    print(
        f"{result['intent']} "
        f"({result['confidence']:.2f}) "
        f"[{result['classification_source']}]"
    )


    print("\nDECISION:")

    print(
        result["decision"]
    )


    print("\nREASON:")

    print(
        result["escalation_reason"]
    )


    print("\nAI REPLY SOURCE:")

    print(
        result.get("reply_source", "unknown")
    )

    print("\nAI REPLY:")

    print(
        result["reply"]
    )


    print("\nHISTORICAL EVIDENCE:")

    if not result["evidence"]:

        print(
            "No evidence found."
        )

    else:

        for i, item in enumerate(
            result["evidence"],
            start=1
        ):

            print(
                f"\n--- Evidence {i} ---"
            )

            print(
                "Customer:",
                clean_text(
                    item["customer_text"]
                )
            )

            print(
                "AmazonHelp:",
                item["brand_text"]
            )

            print(
                f"Similarity: "
                f"{item['similarity']:.4f}"
            )

            print(
                f"Ranking score: "
                f"{item['ranking_score']:.4f}"
            )


    print("\n" + "=" * 70)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    test_queries = [

        "Where is my order? It is late.",

        "My Fire TV is not working.",

        "I want a refund for my order.",

        "Someone hacked my Amazon account.",

        "I cannot place my order."
    ]


    print("\n")

    print("=" * 70)

    print(
        "AMAZONHELP AI SUPPORT AGENT"
    )

    print("=" * 70)


    for query in test_queries:

        result = run_agent(
            query
        )

        print_result(
            result
        )