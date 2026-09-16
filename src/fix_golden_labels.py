import pandas as pd

FILE = "data/golden/amazonhelp_golden.csv"

df = pd.read_csv(FILE, dtype=str, keep_default_na=False)

# Corrections identified during manual review.
corrections = {
    4: ("delivery_issue", "Order whereabouts/tracking is the primary issue."),
    5: ("customer_service", "Complaint about poor customer service."),
    6: ("cancellation_change", "Customer wants another way to cancel an order."),
    7: ("delivery_issue", "Customer has not received the mobile; this is a delivery issue."),
    8: ("delivery_issue", "Order is delayed."),
    9: ("technical_issue", "Customer cannot purchase an Audible book through the app."),
    12: ("delivery_issue", "Package was delivered while customer was away and is now missing."),
}

for index, (intent, note) in corrections.items():
    df.loc[index, "gold_intent"] = intent
    df.loc[index, "label_notes"] = note

df.to_csv(FILE, index=False)

print("Golden labels corrected.")
print()
print(df[df["gold_intent"] != ""][["customer_text", "gold_intent"]].to_string(index=False))
print()
print("Labeled:", (df["gold_intent"] != "").sum())
print("Unlabeled:", (df["gold_intent"] == "").sum())
