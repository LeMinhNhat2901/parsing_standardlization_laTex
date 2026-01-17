"""
Parser module for LaTeX hierarchy extraction
"""
import sys

# CRITICAL: Set recursion limit FIRST before any other imports
# This prevents RecursionError with complex LaTeX structures
if sys.getrecursionlimit() < 10000:
    sys.setrecursionlimit(10000)

# Disable pyparsing packrat to prevent recursion issues
try:
    import pyparsing
    pyparsing.ParserElement.disablePackrat()
except (ImportError, AttributeError):
    pass

from .file_gatherer import FileGatherer
from .hierarchy_builder import HierarchyBuilder
from .latex_cleaner import LaTeXCleaner
from .reference_extractor import ReferenceExtractor
from .deduplicator import Deduplicator

__all__ = [
    'FileGatherer',
    'HierarchyBuilder', 
    'LaTeXCleaner',
    'ReferenceExtractor',
    'Deduplicator'
]