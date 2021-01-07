import re


def clean_with_unders(text: str):
    """
    Deletes . ? - $ ( ) symbols and replaces spaces with under_scores from the text
    Args:
        text (str): string

    Returns:
        string without described symbols
    """
    return text.replace(' ', '_').replace('(', '').replace(')', '').\
        replace('$', '').replace('-', '').replace('.', '').replace('?', '')


def clean_no_unders(text: str):
    """
    Returns a clean string without punctuation and under_scores
    Args:
        text (str): string

    Returns:
        string without described symbols
    """
    return re.sub(r"[^a-zA-Z0-9]+", ' ', text)
