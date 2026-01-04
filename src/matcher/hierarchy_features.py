"""
Extract features from hierarchy.json
These are unique features based on citation context
"""

import json

class HierarchyFeatureExtractor:
    def __init__(self, hierarchy_path):
        """
        Args:
            hierarchy_path: Path to hierarchy.json
        """
        with open(hierarchy_path) as f:
            self.hierarchy = json.load(f)
        
        self.elements = self.hierarchy['elements']
        self.structure = self.hierarchy['hierarchy']
    
    def extract_features(self, bibtex_key):
        """
        Extract hierarchy features for a BibTeX entry
        
        Features:
        - Citation frequency
        - Citation sections
        - Citation depth in hierarchy
        - Co-citation patterns
        - Proximity to figures/tables
        """
        features = {}
        
        # Find where this citation appears
        citation_contexts = self._find_citation_contexts(bibtex_key)
        
        features['citation_count'] = len(citation_contexts)
        features['cited_in_intro'] = self._cited_in_section(citation_contexts, 'introduction')
        features['cited_in_methods'] = self._cited_in_section(citation_contexts, 'methods')
        features['cited_in_results'] = self._cited_in_section(citation_contexts, 'results')
        features['avg_citation_depth'] = self._avg_depth(citation_contexts)
        features['near_figure'] = self._near_figure_or_table(citation_contexts)
        
        return features
    
    def _find_citation_contexts(self, bibtex_key):
        """
        Find all elements containing \cite{bibtex_key}
        
        Returns:
            List of element IDs
        """
        pass
    
    def _cited_in_section(self, contexts, section_name):
        """
        Check if cited in specific section (e.g., Introduction)
        
        Returns:
            Boolean
        """
        pass
    
    def _avg_depth(self, contexts):
        """
        Calculate average depth of citation in hierarchy
        
        Returns:
            Float (average depth level)
        """
        pass
    
    def _near_figure_or_table(self, contexts):
        """
        Check if citation appears near figures/tables
        
        Returns:
            Boolean
        """
        pass
    
    def _get_parent_chain(self, element_id, version):
        """
        Get chain of parents up to root
        
        Returns:
            List of parent IDs
        """
        pass