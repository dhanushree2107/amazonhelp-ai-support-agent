import re


# ============================================================
# V4 HYBRID INTENT RULES
# ============================================================
#
# Main design principle:
# Classify the customer's PRIMARY PROBLEM.
#
# We do not classify based only on secondary words.
#
# Examples:
#
# "My package was not delivered. I want a refund."
#       -> delivery_issue
#
# "I cancelled Prime but was charged."
#       -> payment_billing
#
# "I forgot my password and cannot sign in."
#       -> account_access
#
# ============================================================


def normalize_text(text):
    """Normalize text for rule matching."""
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 1. ACCOUNT ACCESS / SECURITY
# ============================================================

SECURITY_PATTERNS = [
    "hacked",
    "hack my account",
    "account hacked",
    "account compromised",
    "someone hacked",
    "someone changed my email",
    "someone changed my password",
    "changed my email address",
    "changed my password",
    "unauthorized access",
    "unauthorised access",
    "unauthorized",
    "unauthorised",
    "fraud",
    "stolen account",
    "account stolen",
]

ACCOUNT_ACCESS_PATTERNS = [
    "forgot password",
    "forgot my password",
    "forgotten password",
    "forgot password and",
    "cannot sign in",
    "can't sign in",
    "cant sign in",
    "unable to sign in",
    "unable to login",
    "cannot login",
    "can't login",
    "cant login",
    "cannot log in",
    "can't log in",
    "cant log in",
    "unable to log in",
    "login problem",
    "login issue",
    "sign in problem",
    "sign in issue",
    "password reset",
    "reset password",
    "account locked",
    "locked account",
    "account access",
]


# ============================================================
# 2. DELIVERY
# ============================================================

DELIVERY_PATTERNS = [
    "not delivered",
    "not been delivered",
    "wasn't delivered",
    "was not delivered",
    "never delivered",
    "didn't deliver",
    "did not deliver",
    "not arrived",
    "hasn't arrived",
    "hasnt arrived",
    "have not received",
    "haven't received",
    "havent received",
    "didn't receive",
    "did not receive",
    "not received",

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

    "delivery is late",
    "delivery was late",
    "delivery is delayed",
    "delivery was delayed",
    "delivery delayed",

    "package is late",
    "package was late",
    "package is delayed",
    "package was delayed",

    "parcel is late",
    "parcel was late",
    "parcel is delayed",
    "parcel was delayed",

    "still waiting for my order",
    "still waiting for delivery",
    "waiting for my package",
    "waiting for my parcel",

    "supposed to arrive",
    "supposed to be delivered",
    "should have arrived",
    "should have been delivered",

    "shipment delayed",
    "shipment is delayed",
    "nothing was shipped",
    "not shipped",
    "hasn't shipped",
    "hasnt shipped",

    # More specific tracking phrases.
    # We deliberately DO NOT use "tracking" alone.
    "tracking information",
    "tracking number",
    "track my order",
    "track my package",
    "track my parcel",

    "delivery driver",
    "delivery person",
    "delivery depot",

    "package was stolen",
    "parcel was stolen",
    "package stolen",
    "parcel stolen",

    "delivered but",
    "marked delivered",
]


# ============================================================
# 3. WRONG / DAMAGED ITEM
# ============================================================

DAMAGED_PATTERNS = [
    "damaged item",
    "item damaged",
    "arrived damaged",
    "package damaged",
    "parcel damaged",
    "damaged package",
    "damaged parcel",
    "broken item",
    "item is broken",
    "item was broken",
    "broken product",
    "wrong item",
    "wrong product",
    "defective item",
    "defective product",
    "damaged goods",
]


# ============================================================
# 4. PAYMENT / BILLING
# ============================================================

PAYMENT_PATTERNS = [
    "payment failed",
    "payment problem",
    "payment issue",
    "payment method",
    "payment option",
    "payment information",
    "charged",
    "wrong charge",
    "unexpected charge",
    "billing",
    "credit card",
    "debit card",
    "card charged",
    "charged my card",
    "payment declined",

    # Prime billing/payment situations
    "charged for prime",
    "charged for amazon prime",
    "prime charge",
    "prime fee",
    "prime payment",
    "charged after cancelling prime",
    "charged after canceling prime",
    "charged after cancellation",
]


# ============================================================
# 5. REFUND / RETURN
# ============================================================

REFUND_PATTERNS = [
    "refund",
    "money back",
    "reimburse",
    "reimbursement",
    "return my money",
    "refund status",
    "refund hasn't",
    "refund hasnt",
    "refund not",
    "returned item",
    "return item",
    "return my item",
    "return status",
]


# ============================================================
# 6. CANCELLATION / CHANGE
# ============================================================

CANCELLATION_PATTERNS = [
    "cancel my order",
    "cancel order",
    "cancel the order",
    "want to cancel",
    "need to cancel",
    "how to cancel",
    "cancel it",
    "change my order",
    "change the order",
    "modify my order",
    "modify the order",
]


# ============================================================
# 7. TECHNICAL
# ============================================================

TECHNICAL_PATTERNS = [
    "not working",
    "doesn't work",
    "doesnt work",
    "does not work",

    "can't play",
    "cannot play",

    "can't open",
    "cannot open",

    "app problem",
    "app issue",
    "app error",

    "website problem",
    "website error",
    "website not working",

    "fire tv",
    "firetv",
    "kindle",
    "alexa",
    "audible",
    "prime video",

    "technical problem",
    "technical issue",
]


# ============================================================
# 8. ORDER PLACEMENT
# ============================================================

ORDER_PATTERNS = [
    "place an order",
    "place my order",
    "place order",

    "unable to place order",
    "unable to place an order",

    "can't place order",
    "cannot place order",

    "can't place an order",
    "cannot place an order",

    "how to place an order",

    "order something",
]


# ============================================================
# 9. CUSTOMER SERVICE
# ============================================================

CUSTOMER_SERVICE_PATTERNS = [
    "customer service",
    "customer support",
    "support team",
    "support agent",
    "support representative",
    "customer care",

    "speak to someone",
    "talk to someone",

    "contact support",
    "contact customer service",

    "complaint",
    "complain",

    "worst support",
    "bad support",
    "terrible support",
    "poor customer service",
]


# ============================================================
# 10. PRIME MEMBERSHIP
# ============================================================

PRIME_PATTERNS = [
    "amazon prime",
    "prime membership",
    "prime member",
    "prime subscription",
    "prime benefits",

    "signed up for prime",
    "sign up for prime",

    # Membership-specific questions.
    "join prime",
    "cancel prime membership",
    "cancel prime subscription",
    "prime trial",
    "prime free trial",
]


# ============================================================
# HELPER
# ============================================================

def contains_pattern(text, patterns):
    """Return True if any pattern occurs in the text."""
    return any(pattern in text for pattern in patterns)


# ============================================================
# MAIN CLASSIFIER
# ============================================================

def rule_based_intent(text):
    """
    V4 primary-intent rule classifier.

    Returns:
        (intent, confidence)

    If no strong rule matches:
        (None, 0.0)
    """

    text = normalize_text(text)

    if not text:
        return None, 0.0


    # ========================================================
    # 1. SECURITY — ALWAYS HIGHEST PRIORITY
    # ========================================================

    if contains_pattern(text, SECURITY_PATTERNS):
        return "account_access", 0.99


    # ========================================================
    # 2. ACCOUNT ACCESS
    # ========================================================

    if contains_pattern(text, ACCOUNT_ACCESS_PATTERNS):
        return "account_access", 0.96


    # ========================================================
    # 3. DELIVERY
    # ========================================================
    #
    # Delivery gets priority over a secondary refund request.
    #
    # Example:
    # "My package wasn't delivered. I want a refund."
    #
    # -> delivery_issue
    #
    # But if the message is specifically about returning/
    # tracking a returned item, it should not be treated as
    # normal delivery.
    # ========================================================

    if contains_pattern(text, DELIVERY_PATTERNS):

        # Clear damaged-item situation
        if contains_pattern(text, DAMAGED_PATTERNS):
            return "wrong_damaged_item", 0.97

        return "delivery_issue", 0.98


    # ========================================================
    # 4. WRONG / DAMAGED ITEM
    # ========================================================

    if contains_pattern(text, DAMAGED_PATTERNS):
        return "wrong_damaged_item", 0.97


    # ========================================================
    # 5. PAYMENT / BILLING
    # ========================================================
    #
    # Payment takes priority over generic Prime membership
    # when the actual issue is a charge/payment.
    #
    # Example:
    # "I cancelled Prime but was charged today."
    #
    # -> payment_billing
    # ========================================================

    if contains_pattern(text, PAYMENT_PATTERNS):
        return "payment_billing", 0.96


    # ========================================================
    # 6. REFUND / RETURN
    # ========================================================

    if contains_pattern(text, REFUND_PATTERNS):
        return "return_refund", 0.96


    # ========================================================
    # 7. CANCELLATION / CHANGE
    # ========================================================

    if contains_pattern(text, CANCELLATION_PATTERNS):
        return "cancellation_change", 0.96


    # ========================================================
    # 8. TECHNICAL
    # ========================================================

    if contains_pattern(text, TECHNICAL_PATTERNS):
        return "technical_issue", 0.96


    # ========================================================
    # 9. ORDER PLACEMENT
    # ========================================================

    if contains_pattern(text, ORDER_PATTERNS):
        return "order_issue", 0.95


    # ========================================================
    # 10. CUSTOMER SERVICE
    # ========================================================

    if contains_pattern(text, CUSTOMER_SERVICE_PATTERNS):
        return "customer_service", 0.95


    # ========================================================
    # 11. PRIME MEMBERSHIP
    # ========================================================

    if contains_pattern(text, PRIME_PATTERNS):
        return "prime_membership", 0.94


    # ========================================================
    # NO STRONG RULE
    # ========================================================

    return None, 0.0