"""
Unit tests for matcher module
"""

import pytest
from src.matcher import *

def test_data_preparation():
    """Test pair creation"""
    data_prep = DataPreparation()
    
    bibtex_entries = {
        'lipton2018': {'title': 'The mythos', 'authors': ['Lipton']}
    }
    
    references = {
        '1606-03490': {'paper_title': 'The mythos', 'authors': ['Lipton']}
    }
    
    pairs = data_prep.create_pairs_for_publication(bibtex_entries, references)
    
    assert len(pairs) == 1
    assert pairs[0]['bibtex_key'] == 'lipton2018'
    assert pairs[0]['candidate_id'] == '1606-03490'

def test_labeler():
    """Test automatic labeling"""
    labeler = Labeler()
    
    pair = {
        'bibtex_data': {
            'title': 'The mythos of model interpretability',
            'authors': ['Zachary Lipton']
        },
        'candidate_data': {
            'paper_title': 'The Mythos of Model Interpretability',
            'authors': ['Zachary Chase Lipton']
        }
    }
    
    label = labeler.automatic_label(pair)
    assert label == 1  # Should match

def test_feature_extractor():
    """Test feature extraction"""
    extractor = FeatureExtractor()
    
    pair = {
        'bibtex_data': {
            'title': 'Machine learning',
            'authors': ['John Doe'],
            'year': 2020
        },
        'candidate_data': {
            'paper_title': 'Machine Learning',
            'authors': ['John Doe'],
            'submission_date': '2020-01-01'
        }
    }
    
    features = extractor.extract_features(pair)
    
    assert 'title_jaccard' in features
    assert 'author_overlap' in features
    assert 'year_diff' in features

def test_evaluator():
    """Test MRR calculation"""
    evaluator = Evaluator()
    
    predictions = {
        'ref1': ['1606-03490', '1811-10154', '1705-08807'],
        'ref2': ['1811-10154', '1606-03490', '1705-08807']
    }
    
    ground_truth = {
        'ref1': '1606-03490',  # Rank 1 → RR = 1.0
        'ref2': '1606-03490'   # Rank 2 → RR = 0.5
    }
    
    mrr = evaluator.calculate_mrr(predictions, ground_truth)
    assert mrr == 0.75  # (1.0 + 0.5) / 2

if __name__ == '__main__':
    pytest.main([__file__])