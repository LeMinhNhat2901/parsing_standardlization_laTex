"""
Extract traditional text-based features
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=500)
    
    def extract_features(self, pair):
        """
        Extract all features for one pair
        
        Args:
            pair: {bibtex_data, candidate_data, ...}
            
        Returns:
            Feature dict
        """
        features = {}
        
        # Title features
        features.update(self._title_features(pair))
        
        # Author features
        features.update(self._author_features(pair))
        
        # Year features
        features.update(self._year_features(pair))
        
        # Text features
        features.update(self._text_features(pair))
        
        return features
    
    def _title_features(self, pair):
        """
        Title similarity features:
        - Jaccard similarity
        - Levenshtein ratio
        - Cosine similarity (TF-IDF)
        - Exact match
        """
        pass
    
    def _author_features(self, pair):
        """
        Author matching features:
        - Author overlap ratio
        - First author match
        - Last author match
        - Number of common authors
        """
        pass
    
    def _year_features(self, pair):
        """
        Year-based features:
        - Year difference
        - Year exact match
        - Year within 1 year
        """
        pass
    
    def _text_features(self, pair):
        """
        Advanced text features:
        - TF-IDF cosine similarity
        - Word overlap
        - N-gram similarity
        """
        pass
    
    def extract_batch(self, pairs):
        """
        Extract features for multiple pairs
        
        Returns:
            DataFrame with all features
        """
        pass