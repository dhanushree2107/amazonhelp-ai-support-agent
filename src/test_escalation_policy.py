def decide_escalation(
    message,
    intent,
    confidence,
    top_similarity,
    evidence_count
):
    message_lower = message.lower()

    # 1. Security-sensitive issues
    security_keywords = [
        "hacked",
        "hack",
        "fraud",
        "unauthorized",
        "account compromised",
        "changed my email",
        "changed my password",
        "stolen"
    ]

    for keyword in security_keywords:
        if keyword in message_lower:
            return True, "Security-sensitive issue requires human review."

    # 2. Very low classifier confidence
    if confidence < 0.55:
        return True, "Classifier confidence is below the safe threshold."

    # 3. No historical evidence
    if evidence_count == 0:
        return True, "No relevant historical resolution was found."

    # 4. Weak retrieval evidence
    if top_similarity < 0.20:
        return True, "Historical evidence is too weak to safely draft a response."

    # 5. Refund/payment cases require stronger evidence
    sensitive_intents = {
        "return_refund",
        "payment_billing"
    }

    if intent in sensitive_intents and top_similarity < 0.70:
        return True, "Sensitive financial/refund issue lacks sufficiently strong historical evidence."

    return False, "High enough confidence with relevant historical evidence."


def main():

    tests = [
        {
            "message": "Where is my order? It was supposed to arrive yesterday.",
            "intent": "delivery_issue",
            "confidence": 0.6077,
            "similarity": 0.5425
        },
        {
            "message": "My Fire TV is not working.",
            "intent": "technical_issue",
            "confidence": 0.9822,
            "similarity": 0.5583
        },
        {
            "message": "I want a refund for my order.",
            "intent": "return_refund",
            "confidence": 0.6548,
            "similarity": 0.6716
        },
        {
            "message": "Someone hacked my Amazon account and changed my email.",
            "intent": "account_access",
            "confidence": 0.8539,
            "similarity": 0.6354
        }
    ]

    print("=" * 70)
    print("STEP 17 - ESCALATION POLICY TEST")
    print("=" * 70)

    for test in tests:

        escalate, reason = decide_escalation(
            test["message"],
            test["intent"],
            test["confidence"],
            test["similarity"],
            3
        )

        print("\nMessage:")
        print(test["message"])

        print("\nIntent:")
        print(test["intent"])

        print(f"Confidence: {test['confidence']:.4f}")
        print(f"Similarity: {test['similarity']:.4f}")

        print("\nDecision:")

        if escalate:
            print("ESCALATE_TO_HUMAN")
        else:
            print("AUTO_HANDLE")

        print("Reason:")
        print(reason)


if __name__ == "__main__":
    main()
