"""
Reference matching module using machine learning
"""
import sys

# IMPORTANT: Increase recursion limit to avoid RecursionError
# This is needed because some libraries (rapidfuzz, bibtexparser, pyparsing) use deep recursion
# when processing complex LaTeX files with deeply nested environments
if sys.getrecursionlimit() < 10000:
    sys.setrecursionlimit(10000)

# Disable pyparsing packrat to prevent recursion issues
try:
    import pyparsing
    pyparsing.ParserElement.disablePackrat()
except (ImportError, AttributeError):
    pass  # pyparsing not installed or doesn't have this method

from .data_preparation import DataPreparation
from .labeling import Labeler
from .feature_extractor import FeatureExtractor
from .hierarchy_features import HierarchyFeatureExtractor
from .model_trainer import ModelTrainer
from .evaluator import Evaluator

__all__ = [
    'DataPreparation',
    'Labeler',
    'FeatureExtractor',
    'HierarchyFeatureExtractor',
    'ModelTrainer',
    'Evaluator'
]