"""
Text processing utilities for string similarity and normalization
"""

from fuzzywuzzy import fuzz
import re
import unicodedata
from typing import List, Set


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison
    - Convert to lowercase
    - Remove extra whitespace
    - Remove accents/diacritics
    - Normalize unicode
    
    Args:
        text: Input text
    
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove accents
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def calculate_jaccard(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts
    
    Jaccard = |A ∩ B| / |A ∪ B|
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    if not text1 or not text2:
        return 0.0
    
    # Tokenize
    words1 = set(tokenize_text(text1))
    words2 = set(tokenize_text(text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def calculate_levenshtein(text1: str, text2: str) -> float:
    """
    Calculate Levenshtein (fuzzy) similarity ratio
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity ratio (0.0 to 1.0)
    """
    if not text1 or not text2:
        return 0.0
    
    return fuzz.ratio(text1.lower(), text2.lower()) / 100.0


def calculate_token_sort_ratio(text1: str, text2: str) -> float:
    """
    Calculate fuzzy token sort ratio (handles word reordering)
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity ratio (0.0 to 1.0)
    """
    if not text1 or not text2:
        return 0.0
    
    return fuzz.token_sort_ratio(text1.lower(), text2.lower()) / 100.0


def extract_year(date_string: str) -> int:
    """
    Extract year from various date string formats
    
    Args:
        date_string: Date string (e.g., "2020-01-15", "2020", "January 2020")
    
    Returns:
        Year as integer, or 0 if not found
    """
    if not date_string:
        return 0
    
    # Try to find 4-digit year
    match = re.search(r'\b(19|20)\d{2}\b', str(date_string))
    if match:
        return int(match.group())
    
    return 0


def clean_author_name(name: str) -> str:
    """
    Clean and normalize author name
    - Remove titles (Dr., Prof., etc.)
    - Normalize whitespace
    - Handle different formats (First Last, Last First, Last, F.)
    
    Args:
        name: Author name string
    
    Returns:
        Cleaned author name
    """
    if not name:
        return ""
    
    # Remove common titles
    titles = [
        r'\bDr\.?\s*', r'\bProf\.?\s*', r'\bMr\.?\s*', r'\bMs\.?\s*',
        r'\bMrs\.?\s*', r'\bPhD\.?\s*', r'\bMD\.?\s*'
    ]
    
    cleaned = name
    for title in titles:
        cleaned = re.sub(title, '', cleaned, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned.strip()


def extract_last_name(name: str) -> str:
    """
    Extract last name from author name
    
    Args:
        name: Full author name
    
    Returns:
        Last name
    """
    if not name:
        return ""
    
    cleaned = clean_author_name(name)
    
    # Handle "Last, First" format
    if ',' in cleaned:
        return cleaned.split(',')[0].strip()
    
    # Handle "First Last" format
    parts = cleaned.split()
    if parts:
        return parts[-1]
    
    return cleaned


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words
    
    Args:
        text: Input text
    
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split into words
    tokens = text.split()
    
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 1]
    
    return tokens


def remove_stopwords(tokens: List[str], stopwords: Set[str] = None) -> List[str]:
    """
    Remove stop words from token list
    
    Args:
        tokens: List of tokens
        stopwords: Set of stop words (uses default if None)
    
    Returns:
        Filtered token list
    """
    if stopwords is None:
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
            'that', 'these', 'those', 'it', 'its'
        }
    
    return [t for t in tokens if t.lower() not in stopwords]


def calculate_author_overlap(authors1: List[str], authors2: List[str]) -> float:
    """
    Calculate overlap ratio between two author lists
    
    Args:
        authors1: First list of author names
        authors2: Second list of author names
    
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    if not authors1 or not authors2:
        return 0.0
    
    # Extract and normalize last names
    last_names1 = {extract_last_name(a).lower() for a in authors1}
    last_names2 = {extract_last_name(a).lower() for a in authors2}
    
    # Remove empty strings
    last_names1 = {n for n in last_names1 if n}
    last_names2 = {n for n in last_names2 if n}
    
    if not last_names1 or not last_names2:
        return 0.0
    
    # Calculate Jaccard-like overlap
    intersection = last_names1.intersection(last_names2)
    min_size = min(len(last_names1), len(last_names2))
    
    return len(intersection) / min_size


def normalize_latex_text(text: str) -> str:
    """
    Normalize LaTeX text for comparison
    - Remove LaTeX commands
    - Remove math delimiters
    - Normalize whitespace
    
    Args:
        text: LaTeX text
    
    Returns:
        Normalized plain text
    """
    if not text:
        return ""
    
    # Remove common LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
    
    # Remove math delimiters
    text = re.sub(r'\$+', '', text)
    text = re.sub(r'\\[\[\]]', '', text)
    
    # Remove braces
    text = re.sub(r'[{}]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()