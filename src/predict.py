"""
predict.py
----------
Loads the saved category and priority models and classifies new support
tickets from raw text. No dataset or training required — just the two
.pkl files produced by train.py.
"""

import os
import joblib

from preprocess import clean_text

CATEGORY_MODEL_PATH = os.path.join('model', 'category_model.pkl')
PRIORITY_MODEL_PATH = os.path.join('model', 'priority_model.pkl')

# Priority badge displayed in the results table
PRIORITY_BADGE = {
    'Critical' : '🔴  Critical',
    'High'     : '🟠  High',
    'Medium'   : '🟡  Medium',
    'Low'      : '🟢  Low',
}

# LOADING MODELS
def load_models():
    #Load both saved pipelines from disk.

    for path in [CATEGORY_MODEL_PATH, PRIORITY_MODEL_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n  Model file not found: {path}"
                f"\n  Run train.py first to generate the model files."
            )

    cat_model = joblib.load(CATEGORY_MODEL_PATH)
    pri_model = joblib.load(PRIORITY_MODEL_PATH)

    return cat_model, pri_model

# CLASSIFY FUNCTION

def classify_ticket(text: str, cat_model, pri_model) -> tuple:
    
    #Classify a single raw support ticket into a category and priority.

    cleaned  = clean_text(text)
    category = cat_model.predict([cleaned])[0]
    priority = pri_model.predict([cleaned])[0]

    return category, priority

# PRINT RESULTS TABLE

def print_results(tickets: list, cat_model, pri_model) -> None:
    
    #Classify a list of ticket strings and print a formatted results table.

    # Column widths
    col_ticket   = 52
    col_category = 22
    col_priority = 16

    # Header
    print()
    print("  " + "─" * (col_ticket + col_category + col_priority + 8))
    print(
        f"  {'Ticket':<{col_ticket}}"
        f"  {'Category':<{col_category}}"
        f"  {'Priority':<{col_priority}}"
    )
    print("  " + "─" * (col_ticket + col_category + col_priority + 8))

    for ticket in tickets:
        category, priority = classify_ticket(ticket, cat_model, pri_model)

        # Truncate long tickets for display
        display_text = ticket if len(ticket) <= col_ticket else ticket[:col_ticket - 3] + '...'
        badge = PRIORITY_BADGE.get(priority, priority)

        print(
            f"  {display_text:<{col_ticket}}"
            f"  {category:<{col_category}}"
            f"  {badge:<{col_priority}}"
        )

    print("  " + "─" * (col_ticket + col_category + col_priority + 8))
    print()

# EXAMPLE TICKETS

EXAMPLE_TICKETS = [
    # Billing — High
    "I was charged $299 twice this month and need an immediate refund.",

    # Technical — High
    "The entire platform is down and our team cannot log in at all. "
    "We have a client demo in 30 minutes. Please help urgently.",

    # Account — High
    "My account was hacked and the password was changed without my consent. "
    "I cannot access any of my data.",

    # Technical — Medium
    "File uploads are failing intermittently for files larger than 10MB. "
    "It happens about half the time.",

    # Billing — Medium
    "My promo code was not applied at checkout. "
    "The invoice shows full price instead of the discounted rate.",

    # Account — Medium
    "I need to transfer ownership of my account to a colleague "
    "before I leave the company next week.",

    # General — Low
    "Just a quick suggestion — it would be great to have a dark mode "
    "toggle in the settings menu.",

    # Technical — Low
    "Minor issue: the sidebar occasionally collapses on its own "
    "when I resize the browser window.",

    # General — Medium
    "Can you walk me through how to set up the Slack integration? "
    "I couldn't find clear documentation.",

    # Billing — Low
    "Quick question — when does my billing cycle reset each month?",
]

# MAIN

def main():

    print("\n")
    print("  SUPPORT TICKET CLASSIFIER — PREDICTIONS")

    # Load models
    print("\n  Loading models...")
    cat_model, pri_model = load_models()
    print(f"{CATEGORY_MODEL_PATH}")
    print(f"{PRIORITY_MODEL_PATH}")

    # Run predictions on example tickets
    print("\nClassifying example tickets...\n")
    print_results(EXAMPLE_TICKETS, cat_model, pri_model)

    # Interactive mode — let the user type their own ticket
    print("Try your own ticket")
    print("Type a support ticket and press Enter.")
    print("Press Enter on an empty line to exit.\n")

    while True:
        try:
            user_input = input("Ticket: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handles non-interactive environments (CI, piped input)
            break

        if not user_input:
            break

        category, priority = classify_ticket(user_input, cat_model, pri_model)
        badge = PRIORITY_BADGE.get(priority, priority)

        print(f"\nCategory : {category}")
        print(f"Priority : {badge}")
        print()

    print("Done.\n")


if __name__ == '__main__':
    main()