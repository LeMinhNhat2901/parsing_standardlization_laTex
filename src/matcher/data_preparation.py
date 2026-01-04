"""
Prepare data for machine learning
Create all possible (BibTeX, Candidate) pairs
"""

class DataPreparation:
    def __init__(self):
        self.pairs = []
    
    def create_pairs_for_publication(self, bibtex_entries, references_json):
        """
        Create m × n pairs for one publication
        
        Args:
            bibtex_entries: Dict from refs.bib
            references_json: Dict from references.json
            
        Returns:
            List of pair dicts:
            [
                {
                    'bibtex_key': 'lipton2018mythos',
                    'bibtex_data': {...},
                    'candidate_id': '1606-03490',
                    'candidate_data': {...},
                    'publication_id': '2504-13946'
                },
                ...
            ]
        """
        pass
    
    def create_all_pairs(self, publications_dir):
        """
        Create pairs for all publications
        
        Returns:
            List of all pairs across all publications
        """
        pass
    
    def split_data(self, pairs, manual_pubs, auto_pubs):
        """
        Split into train/val/test sets
        
        Args:
            pairs: All pairs
            manual_pubs: List of manually labeled publication IDs
            auto_pubs: List of automatically labeled publication IDs
            
        Returns:
            train_pairs, val_pairs, test_pairs
        """
        pass