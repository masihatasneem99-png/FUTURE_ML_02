"""
Text cleaning and preprocessing utilities for the support ticket classifier.
Used by: train.py, evaluate.py, predict.py
"""

import re
import pandas as pd

# STOPWORD LIST
# General English function words that carry little meaning for classification. This is a simple static list; in production, consider using a more comprehensive list or a library like NLTK's stopwords.

STOPWORDS = set("""
a about after all also an and any are as at
be been being both before but
can could
dear did do does down
each every
for from
has have had he her here him his
i if in is it its
just
kind
me may might more most my
no nor not
of on or our out over
please
quite
really regards
shall she should so some such
thanks that the their them then there these they this those to
under up
very
was we were when while will with would
was were
yet you your
hi hello
""".split())

# CORE CLEANING FUNCTION

def clean_text(text: str) -> str:
    """
    Clean and normalise a raw support ticket string.
    """

    #Convert to string (handles NaN / float inputs safely)
    text = str(text)

    #Lowercase everything
    text = text.lower()

    #  Normalise price tokens  e.g. $149 → PRICE
    text = re.sub(r'\$[\d,]+', 'PRICE', text)

    # Normalise number tokens e.g. 500 → NUM
    text = re.sub(r'\b\d+\b', 'NUM', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', ' ', text)

    #Tokenise by whitespace
    tokens = text.split()

    #Remove stopwords
    #Remove very short tokens (length ≤ 2)
    tokens = [
        token for token in tokens
        if token not in STOPWORDS and len(token) > 2
    ]

    return ' '.join(tokens)

# DATAFRAME-LEVEL HELPER

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full preprocessing to a raw support ticket DataFrame.
    takes a messy, raw data export (specifically from a customer support ticket dataset)
    and shapes it into a clean, structured table that a machine learning model can actually understand.
    Expects the Kaggle customer support ticket dataset with at minimum:
        - 'Ticket Subject'     (str)
        - 'Ticket Description' (str)
        - 'Ticket Type'        (str)  — category label
        - 'Ticket Priority'    (str)  — priority label

    """

    df = df.copy()

    # Drops rows where labels or text are missing
    required_columns = [
        'Ticket Subject',
        'Ticket Description',
        'Ticket Type',
        'Ticket Priority'
    ]
    df = df.dropna(subset=required_columns)

    # Combines 'Ticket Subject' and 'Ticket Description' into 'text'
    df['text'] = (
        df['Ticket Subject'].fillna('').str.strip() + ' ' + df['Ticket Description'].fillna('').str.strip())

    # Applies clean_text() to produce 'clean_text'
    df['clean_text'] = df['text'].apply(clean_text)

    # Drop rows where cleaning produced empty strings
    df = df[df['clean_text'].str.strip() != '']

    # Renames label columns to 'category' and 'priority' for clarity
    df['category'] = df['Ticket Type']
    df['priority']  = df['Ticket Priority']

    return df.reset_index(drop=True)

# QUICK SELF-TEST(runs directly)

if __name__ == '__main__':
    print("  preprocess.py — self-test")

    test_cases = [
        (
            "I was charged $149 twice this month! Need a refund ASAP.",
            "billing / high-priority — price normalisation"
        ),
        (
            "API returning 500 errors since the last deployment. Urgent.",
            "technical / high-priority — number normalisation"
        ),
        (
            "Hi, just wondering when my billing cycle resets each month?",
            "billing / low-priority — salutation removal"
        ),
        (
            "My account was hacked. Password changed without my consent!",
            "account / high-priority — punctuation removal"
        ),
        (
            "Minor suggestion: dark mode toggle in settings would be nice.",
            "general / low-priority — stopword filtering"
        ),
        (
            None,
            "edge case — None input"
        ),
        (
            "   ",
            "edge case — whitespace-only input"
        ),
    ]

    for raw, description in test_cases:
        cleaned = clean_text(raw)
        print(f"\n  [{description}]")
        print(f"Input : {str(raw)[:70]}")
        print(f"Output : {cleaned[:70]}")

    print("  Token count check")

    sample = "I was charged $299 twice and need an immediate refund. The invoice is wrong."
    tokens_before = sample.split()
    tokens_after  = clean_text(sample).split()
    print(f"\nOriginal: {len(tokens_before)} tokens → {sample[:60]}")
    print(f"Cleaned: {len(tokens_after)} tokens → {clean_text(sample)}")
    print(f"Reduction: {len(tokens_before) - len(tokens_after)} tokens removed")

    print("\npreprocess.py is working correctly.\n")