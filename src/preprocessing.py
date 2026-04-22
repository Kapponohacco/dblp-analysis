"""
Text Preprocessing Utilities
==============================
Functions for cleaning and normalizing text data from DBLP records.
"""

import re
from collections import Counter
from typing import Optional

import nltk
from nltk.corpus import stopwords


def ensure_nltk_data():
    """Download required NLTK data if not present."""
    for resource in ["stopwords", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


# Custom stopwords to remove generic scientific filler words.
EXTRA_STOPWORDS = {
    "using", "based", "approach", "towards", "via", "without",
    "two", "one", "three", "case", "use", "used", "also",
    "paper", "study", "results", "proposed", "proposes",
    "method", "methods", "system", "systems", "problem", 
    "model", "models", "new", "novel", "improved", "efficient"
}


def get_stopwords() -> set:
    """Get combined stopword set."""
    ensure_nltk_data()
    return set(stopwords.words("english")) | EXTRA_STOPWORDS


def get_top_keywords(
    titles: list[str],
    top_n: int = 30,
    ngram: int = 1,
    remove_stopwords: bool = True,
) -> list[tuple[str, int]]:
    """
    Extract most frequent keywords from a list of titles.
    """
    counter = Counter()
    sw = get_stopwords() if remove_stopwords else set()

    for title in titles:
        if not title or not isinstance(title, str):
            continue
            
        # Fast cleaning
        text = title.lower()
        text = re.sub(r'[^\w\s-]', ' ', text).replace('-', ' ')
        tokens = [t for t in text.split() if len(t) > 1]

        if ngram == 1:
            if remove_stopwords:
                tokens = [t for t in tokens if t not in sw]
            counter.update(tokens)
        else:
            # Create valid n-grams only if NO word in the phrase is a stopword.
            # This prevents "fake bigrams" (e.g. "applications of learning" -> "applications learning")
            grams = []
            for i in range(len(tokens) - ngram + 1):
                gram_tokens = tokens[i:i+ngram]
                if remove_stopwords and any(t in sw for t in gram_tokens):
                    continue
                grams.append(" ".join(gram_tokens))
            counter.update(grams)

    return counter.most_common(top_n)


def normalize_venue(venue: Optional[str]) -> str:
    """Normalize venue names for consistency."""
    if not venue or not isinstance(venue, str):
        return "Unknown"

    venue = venue.strip()

    # Common abbreviation mappings
    venue_map = {
        "CoRR": "arXiv",
        "CORR": "arXiv",
    }

    return venue_map.get(venue, venue)


def normalize_author(author: str) -> str:
    """
    Normalize author name for consistency.
    Handles common variations (e.g., numbering suffixes).
    """
    if not author:
        return ""

    # Remove DBLP disambiguation suffixes like "0001", "0002"
    author = re.sub(r'\s+\d{4}$', '', author.strip())

    return author.strip()
