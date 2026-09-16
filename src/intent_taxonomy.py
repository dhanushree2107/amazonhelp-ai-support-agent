INTENTS = {
    "delivery_issue": {
        "description": "Problems with delivery, late delivery, missing delivery, tracking, or delivery location.",
        "examples": [
            "My package is late",
            "Tracking says delivered but I didn't receive it",
            "Where is my order?"
        ]
    },

    "order_issue": {
        "description": "General problems with placing, viewing, or managing an order.",
        "examples": [
            "Why can't I place my order?",
            "Can I check my order status?",
            "Something is wrong with my order"
        ]
    },

    "return_refund": {
        "description": "Returns, refunds, refund delays, or requesting money back.",
        "examples": [
            "I want to return this",
            "Where is my refund?",
            "I need a refund"
        ]
    },

    "wrong_damaged_item": {
        "description": "Wrong, damaged, broken, missing, or incorrect item received.",
        "examples": [
            "I received the wrong item",
            "My product arrived damaged",
            "The box was empty"
        ]
    },

    "payment_billing": {
        "description": "Payment problems, unexpected charges, duplicate charges, or billing issues.",
        "examples": [
            "Why was I charged twice?",
            "My payment failed",
            "I was charged unexpectedly"
        ]
    },

    "prime_membership": {
        "description": "Amazon Prime membership, Prime benefits, Prime charges, or Prime subscription issues.",
        "examples": [
            "How do I cancel Prime?",
            "Why was I charged for Prime?",
            "My Prime benefits aren't working"
        ]
    },

    "account_access": {
        "description": "Problems logging in, locked accounts, forgotten account details, or account access.",
        "examples": [
            "My account is locked",
            "I can't log in",
            "I forgot my account email"
        ]
    },

    "technical_issue": {
        "description": "Technical problems with Amazon devices, apps, Prime Video, Kindle, Echo, Fire TV, or similar services.",
        "examples": [
            "My Fire TV isn't working",
            "Prime Video keeps buffering",
            "Alexa isn't responding"
        ]
    },

    "cancellation_change": {
        "description": "Cancelling or changing an existing order or purchase.",
        "examples": [
            "I want to cancel my order",
            "Can I change my order?",
            "I need to cancel a purchase"
        ]
    },

    "customer_service": {
        "description": "Requests to contact customer support or complaints about customer service or escalation.",
        "examples": [
            "How can I contact customer service?",
            "I need to speak to an agent",
            "Your support team hasn't helped me"
        ]
    },

    "other": {
        "description": "Messages that are unclear, non-actionable, unrelated, or do not fit another intent.",
        "examples": [
            "Thanks",
            "Hello",
            "This is ridiculous"
        ]
    }
}


if __name__ == "__main__":
    print("AmazonHelp Intent Taxonomy")
    print("=" * 50)

    for i, (intent, info) in enumerate(INTENTS.items(), 1):
        print(f"{i}. {intent}")
        print(f"   {info['description']}")
