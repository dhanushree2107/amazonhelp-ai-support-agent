import pandas as pd
from pathlib import Path

INPUT_FILE = "data/golden/golden_candidates_200.csv"
OUTPUT_FILE = "data/golden/amazonhelp_golden.csv"

INTENTS = [
    "delivery_issue",
    "order_issue",
    "return_refund",
    "wrong_damaged_item",
    "payment_billing",
    "prime_membership",
    "account_access",
    "technical_issue",
    "cancellation_change",
    "customer_service",
    "other",
]


def show_menu():
    print()

    for i, intent in enumerate(INTENTS, 1):
        print(f"{i:2d}. {intent}")


def load_csv(file_path):
    """
    Load CSV with every column as string.

    This prevents pandas from converting empty columns
    such as label_notes into float64.
    """
    return pd.read_csv(
        file_path,
        dtype=str,
        keep_default_na=False
    )


def main():

    print("=" * 70)
    print("STEP 12.3 - GOLDEN SET MANUAL LABELING")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load input dataset
    # ---------------------------------------------------------
    df = load_csv(INPUT_FILE)

    # Make sure required labeling columns exist
    if "gold_intent" not in df.columns:
        df["gold_intent"] = ""

    if "label_notes" not in df.columns:
        df["label_notes"] = ""

    # ---------------------------------------------------------
    # Continue from existing progress
    # ---------------------------------------------------------
    if Path(OUTPUT_FILE).exists():

        df = load_csv(OUTPUT_FILE)

        # Make sure columns still exist
        if "gold_intent" not in df.columns:
            df["gold_intent"] = ""

        if "label_notes" not in df.columns:
            df["label_notes"] = ""

        print("Continuing existing labeling progress.")

    # ---------------------------------------------------------
    # Force labeling columns to string
    # ---------------------------------------------------------
    df["gold_intent"] = (
        df["gold_intent"]
        .fillna("")
        .astype(str)
    )

    df["label_notes"] = (
        df["label_notes"]
        .fillna("")
        .astype(str)
    )

    total = len(df)

    # ---------------------------------------------------------
    # Label examples
    # ---------------------------------------------------------
    for index in range(total):

        # Skip already labelled rows
        existing = str(
            df.loc[index, "gold_intent"]
        ).strip()

        if existing in INTENTS:
            continue

        print()
        print("=" * 70)
        print(f"Example {index + 1} / {total}")
        print("=" * 70)

        # -----------------------------------------------------
        # Customer message
        # -----------------------------------------------------
        print()
        print("CUSTOMER MESSAGE:")
        print("-" * 70)
        print(df.loc[index, "customer_text"])
        print("-" * 70)

        # -----------------------------------------------------
        # Cleaned message
        # -----------------------------------------------------
        print()
        print("CLEANED MESSAGE:")
        print("-" * 70)
        print(df.loc[index, "customer_text_clean"])
        print("-" * 70)

        # -----------------------------------------------------
        # Historical AmazonHelp reply
        # -----------------------------------------------------
        print()
        print("HISTORICAL AMAZONHELP REPLY:")
        print("-" * 70)
        print(df.loc[index, "brand_text"])
        print("-" * 70)

        # -----------------------------------------------------
        # Candidate suggestion
        # -----------------------------------------------------
        print()
        print(
            "Candidate suggestion:",
            df.loc[index, "candidate_intent"]
        )

        # -----------------------------------------------------
        # Intent menu
        # -----------------------------------------------------
        show_menu()

        # -----------------------------------------------------
        # Ask for label
        # -----------------------------------------------------
        while True:

            choice = input(
                "\nYour label (1-11, q=quit): "
            ).strip()

            # -------------------------------------------------
            # Quit and save progress
            # -------------------------------------------------
            if choice.lower() == "q":

                df.to_csv(
                    OUTPUT_FILE,
                    index=False
                )

                print()
                print("Progress saved.")
                print(f"Output: {OUTPUT_FILE}")

                return

            # -------------------------------------------------
            # Validate numeric input
            # -------------------------------------------------
            if choice.isdigit():

                number = int(choice)

                if 1 <= number <= len(INTENTS):

                    intent = INTENTS[number - 1]

                    # Save gold intent
                    df.loc[index, "gold_intent"] = intent

                    # Ask for optional note
                    note = input(
                        "Optional note (press Enter to skip): "
                    ).strip()

                    # Save note
                    df.loc[index, "label_notes"] = note

                    # Save immediately after every label
                    df.to_csv(
                        OUTPUT_FILE,
                        index=False
                    )

                    print()
                    print(f"Saved: {intent}")

                    break

            print(
                "Invalid choice. Enter 1-11 or q."
            )

    # ---------------------------------------------------------
    # Completed
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("GOLDEN SET LABELING COMPLETE")
    print("=" * 70)
    print(f"Total examples: {total}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()