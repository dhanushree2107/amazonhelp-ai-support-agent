import pandas as pd

FILE = "data/golden/amazonhelp_golden.csv"

df = pd.read_csv(
    FILE,
    dtype=str,
    keep_default_na=False
)

# Corrections are based on the primary user problem.
# Index = example number - 1
corrections = {

    # Example 20
    19: (
        "prime_membership",
        "Customer forgot about Prime subscription and was charged for it."
    ),

    # Example 22
    21: (
        "customer_service",
        "Primary complaint is poor customer service and unresolved refund handling."
    ),

    # Example 27
    26: (
        "technical_issue",
        "Echo Dot firmware/device functionality issue."
    ),

    # Example 29
    28: (
        "delivery_issue",
        "Primary issue is packages arriving 4+ days late."
    ),

    # Example 30
    29: (
        "payment_billing",
        "Customer is asking about managing credit cards in Amazon Wallet."
    ),

    # Example 31
    30: (
        "delivery_issue",
        "Primary issue is repeated package delivery delays."
    ),

    # Example 32
    31: (
        "delivery_issue",
        "Customer says Prime orders are arriving later than promised."
    ),

    # Example 33
    32: (
        "delivery_issue",
        "Complaint concerns unsafe package delivery location."
    ),

    # Example 35
    34: (
        "delivery_issue",
        "Replacement was selected but customer has not received tracking information."
    ),

    # Example 36
    35: (
        "delivery_issue",
        "Primary issue is repeated late Prime delivery; refund is secondary."
    ),

    # Example 37
    36: (
        "payment_billing",
        "Customer was charged for a cancelled Prime subscription."
    ),

    # Example 38
    37: (
        "delivery_issue",
        "Customer paid for one-day delivery but order was not delivered."
    ),

    # Example 43
    42: (
        "customer_service",
        "Complaint is specifically about poor customer service/chat quality."
    ),

    # Example 49
    48: (
        "cancellation_change",
        "Customer says Amazon cancelled the purchase/order."
    ),

    # Example 51
    50: (
        "wrong_damaged_item",
        "Product arrived damaged and contents were ruined."
    ),

    # Example 53
    52: (
        "delivery_issue",
        "Primary issue is a package being a week late."
    ),

    # Example 54
    53: (
        "delivery_issue",
        "Customer believes package was stolen and asks about delivery alternatives."
    ),

    # Example 55
    54: (
        "return_refund",
        "Customer is asking about a refund that has not been processed."
    ),

    # Example 56
    55: (
        "payment_billing",
        "Customer cannot use an Indian credit card."
    ),

    # Example 58
    57: (
        "return_refund",
        "Customer asks how long a refund will take."
    ),

    # Example 59
    58: (
        "account_access",
        "Customer cannot log in because their account email was changed."
    ),

    # Example 61
    60: (
        "account_access",
        "Customer has been unable to log in for more than a month."
    ),

    # Example 62
    61: (
        "payment_billing",
        "Customer asks about an unexplained £1 charge."
    ),

    # Example 63
    62: (
        "customer_service",
        "Complaint is about representatives hanging up and poor phone support."
    ),

    # Example 64
    63: (
        "wrong_damaged_item",
        "TV arrived broken; customer discusses return/refund and replacement."
    ),

    # Example 65
    64: (
        "delivery_issue",
        "Primary issue is an order that was not delivered; refund is secondary."
    ),

    # Example 66
    65: (
        "return_refund",
        "Customer requested a return but pickup has not happened."
    ),

    # Example 67
    66: (
        "account_access",
        "Issue involved duplicate accounts using the same email address."
    ),

    # Example 69
    68: (
        "customer_service",
        "Primary complaint is horrible customer service and misleading support."
    ),

    # Example 71
    70: (
        "prime_membership",
        "Customer is dealing with an unwanted Prime subscription/renewal."
    ),

    # Example 73
    72: (
        "delivery_issue",
        "Primary issue is multiple packages not arriving on time."
    ),

    # Example 74
    73: (
        "return_refund",
        "Customer's returned item is taking too long to process."
    ),

    # Example 75
    74: (
        "delivery_issue",
        "Message concerns Prime delivery expectations."
    ),

    # Example 76
    75: (
        "delivery_issue",
        "Package arrived later than the promised delivery time."
    ),

    # Example 77
    76: (
        "wrong_damaged_item",
        "Customer reports multiple products arriving damaged."
    ),

    # Example 81
    80: (
        "prime_membership",
        "Customer says Prime was activated without consent and asks about cancellation."
    ),

    # Example 82
    81: (
        "technical_issue",
        "Customer reports the Amazon app is crashing."
    ),

    # Example 83
    82: (
        "prime_membership",
        "Customer asks about content availability in Prime Video."
    ),

    # Example 84
    83: (
        "account_access",
        "Customer cannot log in despite changing their password."
    ),

    # Example 85
    84: (
        "delivery_issue",
        "Package collection point was changed and customer cannot access it."
    ),

    # Example 86
    85: (
        "wrong_damaged_item",
        "Message praises quick resolution of damaged goods."
    ),

    # Example 88
    87: (
        "delivery_issue",
        "Customer complains about failed Prime delivery and requests a credit."
    ),

    # Example 91
    90: (
        "delivery_issue",
        "Preordered item has not shipped and customer is worried about release-day delivery."
    ),

    # Example 92
    91: (
        "delivery_issue",
        "Account says delivered but customer received nothing."
    ),

    # Example 94
    93: (
        "cancellation_change",
        "Customer accidentally ordered Kindle version and wants to cancel."
    ),

    # Example 96
    95: (
        "customer_service",
        "Customer complains about incompetence/support quality."
    ),

    # Example 97
    96: (
        "customer_service",
        "Customer complains about how support handled repeated returns."
    ),

    # Example 98
    97: (
        "order_issue",
        "Customer asks whether they can send an order number by direct message."
    ),

    # Example 99
    98: (
        "prime_membership",
        "Customer was pushed into Prime and wants to cancel the membership."
    ),

    # Example 102
    101: (
        "technical_issue",
        "Voice search does not search content from other apps."
    ),

    # Example 104
    103: (
        "delivery_issue",
        "Product delivery was cancelled unexpectedly."
    ),

    # Example 105
    104: (
        "payment_billing",
        "Customer repeatedly encounters a payment problem."
    ),

    # Example 108
    107: (
        "account_access",
        "Customer cannot access account because password reset email is not received."
    ),

    # Example 109
    108: (
        "delivery_issue",
        "Package is stuck at the warehouse past the expected delivery time."
    ),

    # Example 111
    110: (
        "cancellation_change",
        "Customer's brother cancelled the order and customer wants help."
    ),

    # Example 112
    111: (
        "delivery_issue",
        "Customer is being told their order will arrive five days late."
    ),

    # Example 115
    114: (
        "delivery_issue",
        "Customer did not receive the order."
    ),

    # Example 116
    115: (
        "customer_service",
        "Complaint is about representatives giving poor and misleading support."
    ),

    # Example 117
    116: (
        "return_refund",
        "Customer asks about returning Amazon purchases to Whole Foods."
    ),

    # Example 119
    118: (
        "delivery_issue",
        "Customer wants to pick up a package before it is returned to seller."
    ),

    # Example 120
    119: (
        "delivery_issue",
        "Preordered item has not been dispatched."
    ),

    # Example 122
    121: (
        "wrong_damaged_item",
        "Two dishwashers arrived damaged."
    ),

    # Example 123
    122: (
        "cancellation_change",
        "Customer's PrimeNow order was cancelled."
    ),

    # Example 124
    123: (
        "wrong_damaged_item",
        "Customer reports an item arrived damaged."
    ),

    # Example 125
    124: (
        "delivery_issue",
        "Return label discussion concerns an item that has not been delivered."
    ),

    # Example 126
    125: (
        "wrong_damaged_item",
        "Customer received the wrong/faulty product."
    ),

    # Example 128
    127: (
        "customer_service",
        "Customer praises fast customer service resolution."
    ),

    # Example 129
    128: (
        "delivery_issue",
        "Customer cannot track a package that is still on its way."
    ),

    # Example 130
    129: (
        "wrong_damaged_item",
        "Customer reports a defective product."
    ),

    # Example 131
    130: (
        "prime_membership",
        "Customer asks about activating Prime membership."
    ),

    # Example 132
    131: (
        "return_refund",
        "Customer received only part of the expected refund."
    ),

    # Example 133
    132: (
        "customer_service",
        "Complaint is about cancellations, blocked accounts, and poor customer treatment."
    ),

    # Example 136
    135: (
        "delivery_issue",
        "Shipment was late and package was reported lost or damaged."
    ),

    # Example 137
    136: (
        "payment_billing",
        "Order repeatedly switches to Payment Revision."
    ),

    # Example 138
    137: (
        "delivery_issue",
        "Items arrived late despite Prime membership."
    ),

    # Example 139
    138: (
        "account_access",
        "Customer forgot password and cannot sign in."
    ),

    # Example 143
    142: (
        "delivery_issue",
        "Customer is asking about the promised pickup date."
    ),

    # Example 144
    143: (
        "payment_billing",
        "Customer was charged for an unwanted Kindle subscription."
    ),

    # Example 145
    144: (
        "delivery_issue",
        "Customer ordered a product that was not delivered."
    ),

    # Example 146
    145: (
        "wrong_damaged_item",
        "Customer reports a defective/dangerous product."
    ),

    # Example 147
    146: (
        "customer_service",
        "Overall complaint about repeated Amazon service problems."
    ),

    # Example 148
    147: (
        "delivery_issue",
        "Orders were cancelled because sellers would not deliver to the customer's area."
    ),

    # Example 149
    148: (
        "customer_service",
        "Customer is asking about how to contact customer service."
    ),

    # Example 150
    149: (
        "customer_service",
        "Customer wants to be connected to customer care."
    ),

    # Example 151
    150: (
        "prime_membership",
        "Customer is joking about being addicted to Prime."
    ),

    # Example 153
    152: (
        "return_refund",
        "Customer is processing a return/replacement."
    ),

    # Example 154
    153: (
        "account_access",
        "Customer's account history and Prime membership disappeared."
    ),

    # Example 155
    154: (
        "payment_billing",
        "App errors occur while making a payment."
    ),

    # Example 156
    155: (
        "wrong_damaged_item",
        "Customer reports a defective tempered glass product."
    ),

    # Example 157
    156: (
        "payment_billing",
        "Customer reports an unexplained debit-card charge."
    ),

    # Example 158
    157: (
        "delivery_issue",
        "Prime package was not delivered."
    ),

    # Example 159
    158: (
        "account_access",
        "Customer's account is locked after suspected fraud."
    ),

    # Example 160
    159: (
        "delivery_issue",
        "Customer complains that UPS deliveries are repeatedly delayed."
    ),

    # Example 161
    160: (
        "delivery_issue",
        "Customer is concerned about how the package was delivered."
    ),

    # Example 162
    161: (
        "technical_issue",
        "Customer reports Kindle/customer-service/content-syncing problems."
    ),

    # Example 163
    162: (
        "delivery_issue",
        "Customer complains about failure of next-day delivery."
    ),

    # Example 164
    163: (
        "customer_service",
        "Customer says they still have no information from support."
    ),

    # Example 165
    164: (
        "payment_billing",
        "Customer's promotional credit is missing from the account."
    ),

    # Example 166
    165: (
        "payment_billing",
        "Customer needs to pay after receiving a previously refunded item."
    ),

    # Example 167
    166: (
        "return_refund",
        "Customer's return tracking/status has not been updated."
    ),

    # Example 168
    167: (
        "prime_membership",
        "Customer complains about lack of benefit from Prime membership."
    ),

    # Example 170
    169: (
        "delivery_issue",
        "Orders were cancelled because they were damaged in transit."
    ),

    # Example 171
    170: (
        "customer_service",
        "Customer complains about support feedback and unresolved service."
    ),

    # Example 172
    171: (
        "cancellation_change",
        "Customer wants to cancel an order but cannot find the option."
    ),

    # Example 173
    172: (
        "delivery_issue",
        "Customer is waiting for a shipment past the promised delivery date."
    ),

    # Example 175
    174: (
        "delivery_issue",
        "Customer has not received an order confirmation and fears cancellation."
    ),

    # Example 176
    175: (
        "delivery_issue",
        "Delivery did not happen because of a holiday."
    ),

    # Example 177
    176: (
        "delivery_issue",
        "Prime delivery is already three days late."
    ),

    # Example 180
    179: (
        "customer_service",
        "Customer complains about customer care repeatedly giving poor information."
    ),

    # Example 181
    180: (
        "return_refund",
        "Customer received a refund despite being promised the order would still arrive."
    ),

    # Example 183
    182: (
        "technical_issue",
        "Prime Video playback is not working for some videos."
    ),

    # Example 185
    184: (
        "customer_service",
        "Customer is reporting that information was sent to customer service."
    ),

    # Example 186
    185: (
        "return_refund",
        "Customer is waiting for a promised refund."
    ),

    # Example 187
    186: (
        "account_access",
        "Customer has an account-sharing/purchase-confirmation issue."
    ),

    # Example 188
    187: (
        "account_access",
        "Customer needs help identifying themselves to fix their account."
    ),

    # Example 189
    188: (
        "payment_billing",
        "Customer is waiting for promised bank-fee/purchase refunds."
    ),

    # Example 190
    189: (
        "customer_service",
        "Complaint is about delivery boys and incompetent call-center support."
    ),

    # Example 191
    190: (
        "customer_service",
        "Customer reports extremely poor support."
    ),

    # Example 193
    192: (
        "customer_service",
        "Customer is complaining about Prime support/service."
    ),

    # Example 194
    193: (
        "other",
        "Very short follow-up with insufficient context to determine a specific intent."
    ),

    # Example 197
    196: (
        "delivery_issue",
        "Customer received a package belonging to someone else."
    ),

    # Example 198
    197: (
        "order_issue",
        "Customer needs a gift receipt and correctly sized gift bag for an order."
    ),
}


changed = 0

for index, (intent, note) in corrections.items():

    if index >= len(df):
        continue

    old_intent = df.loc[index, "gold_intent"]

    if old_intent != intent:
        print(
            f"Example {index + 1}: "
            f"{old_intent} -> {intent}"
        )

        df.loc[index, "gold_intent"] = intent
        df.loc[index, "label_notes"] = note

        changed += 1


df.to_csv(FILE, index=False)

print()
print(f"Corrections applied: {changed}")

print()
print("NEW GOLDEN DISTRIBUTION:")
print(df["gold_intent"].value_counts())

print()
print("Total labelled:", (df["gold_intent"] != "").sum())
