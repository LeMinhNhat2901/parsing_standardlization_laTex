"""
Utility modules for the project
"""
import sys

# CRITICAL: Set recursion limit FIRST before any other imports
# This prevents RecursionError with complex data structures
if sys.getrecursionlimit() < 10000:
    sys.setrecursionlimit(10000)

# Disable pyparsing packrat to prevent recursion issues
try:
    import pyparsing
    pyparsing.ParserElement.disablePackrat()
except (ImportError, AttributeError):
    pass

from .file_io import load_json, save_json, load_bibtex, save_bibtex, ensure_dir

# Import safe type utilities (for avoiding RecursionError)
from .safe_types import (
    safe_type_name, is_string, is_int, is_float, is_bool,
    is_numeric, is_list, is_dict, is_none, is_primitive, is_ndarray,
    to_primitive, to_float_safe, to_int_safe,
    safe_dict_copy, flatten_for_dataframe
)

# Import text_utils functions with error handling
try:
    from .text_utils import (
        normalize_text, calculate_jaccard, calculate_levenshtein,
        extract_year, clean_author_name, tokenize_text
    )
    _TEXT_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some text_utils functions not available: {e}")
    _TEXT_UTILS_AVAILABLE = False
    # Provide fallback implementations
    def normalize_text(text): return str(text).lower().strip() if text else ""
    def calculate_jaccard(t1, t2): return 0.0
    def calculate_levenshtein(t1, t2): return 0
    def extract_year(text): return None
    def clean_author_name(name): return str(name) if name else ""
    def tokenize_text(text): return str(text).split() if text else []

from .logger import setup_logger, get_logger

__all__ = [
    'load_json', 'save_json', 'load_bibtex', 'save_bibtex', 'ensure_dir',
    'safe_type_name', 'is_string', 'is_int', 'is_float', 'is_bool',
    'is_numeric', 'is_list', 'is_dict', 'is_none', 'is_primitive', 'is_ndarray',
    'to_primitive', 'to_float_safe', 'to_int_safe',
    'safe_dict_copy', 'flatten_for_dataframe',
    'normalize_text', 'calculate_jaccard', 'calculate_levenshtein',
    'extract_year', 'clean_author_name', 'tokenize_text',
    'setup_logger', 'get_logger'
]
