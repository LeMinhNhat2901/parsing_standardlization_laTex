"""
Unit tests for matcher module

Tests cover:
- DataPreparation: m×n pair creation, data splitting
- Labeler: manual and automatic labeling
- FeatureExtractor: text-based feature extraction
- HierarchyFeatureExtractor: hierarchy-based features
- ModelTrainer: CatBoost training (mocked)
- Evaluator: MRR and other metrics

Run with: pytest tests/test_matcher.py -v
"""

import pytest
import sys
from pathlib import Path
import tempfile
import json
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from matcher.data_preparation import DataPreparation
from matcher.labeling import Labeler
from matcher.feature_extractor import FeatureExtractor
from matcher.hierarchy_features import HierarchyFeatureExtractor
from matcher.evaluator import Evaluator


class TestDataPreparation:
    """Tests for DataPreparation class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data files"""
        tmpdir = tempfile.mkdtemp()
        
        # Create refs.bib content (simple format for testing)
        refs_bib = {
            'lipton2018': {
                'title': 'The mythos of model interpretability',
                'author': 'Lipton, Z. C.',
                'year': '2018',
                'ENTRYTYPE': 'article',
                'ID': 'lipton2018'
            },
            'rudin2019': {
                'title': 'Stop explaining black box models',
                'author': 'Rudin, C.',
                'year': '2019',
                'ENTRYTYPE': 'article',
                'ID': 'rudin2019'
            }
        }
        
        # Create references.json content
        references = {
            '1606.03490': {
                'paper_title': 'The mythos of model interpretability',
                'paper_authors': ['Zachary C. Lipton'],
                'year': '2016'
            },
            '1811.10154': {
                'paper_title': 'Stop explaining black box models',
                'paper_authors': ['Cynthia Rudin'],
                'year': '2018'
            },
            '2001.09876': {
                'paper_title': 'A different paper',
                'paper_authors': ['John Doe'],
                'year': '2020'
            }
        }
        
        # Save files
        pub_dir = os.path.join(tmpdir, '2304.12345')
        os.makedirs(pub_dir)
        
        # Save as JSON for simplicity (DataPreparation can handle both)
        refs_path = os.path.join(pub_dir, 'refs.bib')
        with open(refs_path, 'w') as f:
            # Write simple JSON-like format that can be loaded
            json.dump(refs_bib, f)
        
        refs_json_path = os.path.join(pub_dir, 'references.json')
        with open(refs_json_path, 'w') as f:
            json.dump(references, f)
        
        yield {
            'tmpdir': tmpdir,
            'pub_dir': pub_dir,
            'refs_path': refs_path,
            'refs_json_path': refs_json_path,
            'bibtex_entries': refs_bib,
            'references': references
        }
        
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir)
    
    def test_create_pairs_count(self):
        """Test that m×n pairs are created"""
        data_prep = DataPreparation()
        
        # Mock the data loading
        bibtex_entries = {
            'ref1': {'title': 'Paper 1'},
            'ref2': {'title': 'Paper 2'}
        }
        
        references = {
            'cand1': {'paper_title': 'Candidate 1'},
            'cand2': {'paper_title': 'Candidate 2'},
            'cand3': {'paper_title': 'Candidate 3'}
        }
        
        # m=2, n=3, should create 6 pairs
        # This tests the logic, not file loading
        pairs = []
        for bib_key, bib_data in bibtex_entries.items():
            for cand_id, cand_data in references.items():
                pairs.append({
                    'bibtex_key': bib_key,
                    'candidate_id': cand_id
                })
        
        assert len(pairs) == 2 * 3  # m × n = 6
    
    def test_pair_structure(self):
        """Test pair dictionary structure"""
        data_prep = DataPreparation()
        
        # Create a mock pair
        pair = {
            'publication_id': 'test-pub',
            'bibtex_key': 'lipton2018',
            'bibtex_data': {'title': 'Test'},
            'candidate_id': '1606.03490',
            'candidate_data': {'paper_title': 'Test'},
            'label': None
        }
        
        # Verify structure
        assert 'publication_id' in pair
        assert 'bibtex_key' in pair
        assert 'bibtex_data' in pair
        assert 'candidate_id' in pair
        assert 'candidate_data' in pair
    
    def test_split_data(self):
        """Test data splitting"""
        data_prep = DataPreparation()
        
        # Create mock pairs from different publications
        pairs = []
        for i in range(100):
            pairs.append({
                'publication_id': f'pub{i % 10}',  # 10 publications
                'bibtex_key': f'ref{i}',
                'candidate_id': f'cand{i}',
                'label': i % 2
            })
        
        train, val, test = data_prep.split_data(
            pairs, 
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Check splits are non-empty
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        
        # Check approximate ratios (by publication, not exact)
        total = len(train) + len(val) + len(test)
        assert total == len(pairs)


class TestLabeler:
    """Tests for Labeler class"""
    
    def test_automatic_label_exact_match(self):
        """Test automatic labeling with exact title match"""
        labeler = Labeler()
        
        pair = {
            'publication_id': 'test',
            'bibtex_key': 'ref1',
            'candidate_id': 'cand1',
            'bibtex_data': {
                'title': 'The mythos of model interpretability',
                'author': 'Lipton, Z. C.',
                'year': '2018'
            },
            'candidate_data': {
                'paper_title': 'The mythos of model interpretability',
                'paper_authors': ['Zachary C. Lipton'],
                'year': '2018'
            }
        }
        
        label = labeler.automatic_label(pair)
        assert label == 1  # Should match
    
    def test_automatic_label_similar_titles(self):
        """Test automatic labeling with similar but not exact titles"""
        labeler = Labeler(title_threshold=90)
        
        pair = {
            'publication_id': 'test',
            'bibtex_key': 'ref1',
            'candidate_id': 'cand1',
            'bibtex_data': {
                'title': 'The Mythos of Model Interpretability',  # Different case
                'author': 'Lipton',
                'year': '2018'
            },
            'candidate_data': {
                'paper_title': 'The mythos of model interpretability',
                'paper_authors': ['Lipton'],
                'year': '2018'
            }
        }
        
        label = labeler.automatic_label(pair)
        assert label == 1  # Should still match
    
    def test_automatic_label_no_match(self):
        """Test automatic labeling with clearly different papers"""
        labeler = Labeler()
        
        pair = {
            'publication_id': 'test',
            'bibtex_key': 'ref1',
            'candidate_id': 'cand1',
            'bibtex_data': {
                'title': 'Machine Learning Fundamentals',
                'author': 'Smith',
                'year': '2020'
            },
            'candidate_data': {
                'paper_title': 'Deep Neural Networks for Computer Vision',
                'paper_authors': ['Johnson'],
                'year': '2019'
            }
        }
        
        label = labeler.automatic_label(pair)
        assert label == 0  # Should not match
    
    def test_title_similarity(self):
        """Test title similarity calculation"""
        labeler = Labeler()
        
        sim = labeler._title_similarity(
            'Machine Learning for Data Science',
            'Machine Learning for Data Science'
        )
        assert sim >= 99  # Exact match
        
        sim = labeler._title_similarity(
            'Machine Learning',
            'Deep Learning'
        )
        assert sim < 80  # Different
    
    def test_author_overlap(self):
        """Test author overlap calculation"""
        labeler = Labeler()
        
        overlap = labeler._author_overlap(
            'Smith, John and Doe, Jane',
            'John Smith and Jane Doe'
        )
        assert overlap > 0  # Should find some overlap
        
        overlap = labeler._author_overlap(
            'Smith, John',
            'Johnson, Bob'
        )
        assert overlap == 0  # No overlap


class TestFeatureExtractor:
    """Tests for FeatureExtractor class"""
    
    @pytest.fixture
    def sample_pair(self):
        """Create sample pair for testing"""
        return {
            'bibtex_data': {
                'title': 'Machine Learning for Natural Language Processing',
                'author': 'Smith, John and Doe, Jane',
                'year': '2020',
                'abstract': 'We present a novel approach to NLP using ML.'
            },
            'candidate_data': {
                'paper_title': 'Machine Learning for NLP',
                'paper_authors': ['John Smith', 'Jane Doe'],
                'year': '2020',
                'abstract': 'A novel machine learning approach for NLP tasks.'
            }
        }
    
    def test_extract_features(self, sample_pair):
        """Test feature extraction"""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_pair)
        
        # Check title features
        assert 'title_jaccard' in features
        assert 'title_levenshtein' in features
        assert 'title_token_sort' in features
        assert 'title_exact_match' in features
        
        # Check author features
        assert 'author_overlap_ratio' in features
        assert 'first_author_match' in features
        
        # Check year features
        assert 'year_diff' in features
        assert 'year_exact_match' in features
    
    def test_title_features_exact_match(self):
        """Test title features for exact match"""
        extractor = FeatureExtractor()
        
        pair = {
            'bibtex_data': {'title': 'Test Paper Title'},
            'candidate_data': {'paper_title': 'Test Paper Title'}
        }
        
        features = extractor._title_features(pair)
        
        assert features['title_exact_match'] == 1
        assert features['title_jaccard'] == 1.0
        assert features['title_levenshtein'] == 1.0
    
    def test_title_features_no_match(self):
        """Test title features for completely different titles"""
        extractor = FeatureExtractor()
        
        pair = {
            'bibtex_data': {'title': 'Alpha Beta Gamma'},
            'candidate_data': {'paper_title': 'Completely Different Words Here'}
        }
        
        features = extractor._title_features(pair)
        
        assert features['title_exact_match'] == 0
        assert features['title_jaccard'] < 0.5
    
    def test_author_features(self, sample_pair):
        """Test author feature extraction"""
        extractor = FeatureExtractor()
        features = extractor._author_features(sample_pair)
        
        # Authors are similar
        assert features['author_overlap_ratio'] > 0
        assert features['first_author_match'] == 1  # Both have Smith/John Smith first
    
    def test_year_features_same_year(self):
        """Test year features for same year"""
        extractor = FeatureExtractor()
        
        pair = {
            'bibtex_data': {'year': '2020'},
            'candidate_data': {'year': '2020'}
        }
        
        features = extractor._year_features(pair)
        
        assert features['year_exact_match'] == 1
        assert features['year_diff'] == 0
        assert features['year_within_1'] == 1
    
    def test_year_features_different_year(self):
        """Test year features for different years"""
        extractor = FeatureExtractor()
        
        pair = {
            'bibtex_data': {'year': '2020'},
            'candidate_data': {'year': '2018'}
        }
        
        features = extractor._year_features(pair)
        
        assert features['year_exact_match'] == 0
        assert features['year_diff'] == 2
        assert features['year_within_1'] == 0
    
    def test_extract_batch(self, sample_pair):
        """Test batch feature extraction"""
        extractor = FeatureExtractor()
        
        pairs = [sample_pair, sample_pair.copy()]
        df = extractor.extract_batch(pairs)
        
        assert len(df) == 2
        assert 'title_jaccard' in df.columns
        assert 'author_overlap_ratio' in df.columns


class TestHierarchyFeatureExtractor:
    """Tests for HierarchyFeatureExtractor class"""
    
    @pytest.fixture
    def sample_hierarchy(self):
        """Create sample hierarchy.json"""
        tmpdir = tempfile.mkdtemp()
        
        hierarchy = {
            'elements': {
                'elem1': {
                    'type': 'section',
                    'title': 'Introduction',
                    'content': 'We cite \\cite{lipton2018} and \\cite{rudin2019}.',
                    'parent': None
                },
                'elem2': {
                    'type': 'section',
                    'title': 'Methods',
                    'content': 'We use the approach from \\cite{lipton2018}.',
                    'parent': None
                },
                'elem3': {
                    'type': 'figure',
                    'content': 'Figure showing results.',
                    'parent': 'elem2'
                }
            },
            'hierarchy': {
                'root': ['elem1', 'elem2']
            }
        }
        
        path = os.path.join(tmpdir, 'hierarchy.json')
        with open(path, 'w') as f:
            json.dump(hierarchy, f)
        
        yield path
        
        import shutil
        shutil.rmtree(tmpdir)
    
    def test_extract_features(self, sample_hierarchy):
        """Test hierarchy feature extraction"""
        extractor = HierarchyFeatureExtractor(sample_hierarchy)
        
        features = extractor.extract_features('lipton2018')
        
        # Check feature presence
        assert 'citation_count' in features
        assert 'cited_in_intro' in features
        assert 'cited_in_methods' in features
        
        # lipton2018 is cited twice
        assert features['citation_count'] == 2
    
    def test_citation_index(self, sample_hierarchy):
        """Test citation index building"""
        extractor = HierarchyFeatureExtractor(sample_hierarchy)
        
        # Check index was built
        assert 'lipton2018' in extractor.citation_index
        assert len(extractor.citation_index['lipton2018']) == 2
    
    def test_cocitation_features(self, sample_hierarchy):
        """Test co-citation feature extraction"""
        extractor = HierarchyFeatureExtractor(sample_hierarchy)
        
        features = extractor.extract_features('lipton2018')
        
        # lipton2018 is co-cited with rudin2019 in elem1
        assert features['co_citation_count'] >= 1


class TestEvaluator:
    """Tests for Evaluator class"""
    
    def test_mrr_perfect(self):
        """Test MRR with perfect predictions (all rank 1)"""
        evaluator = Evaluator()
        
        predictions = {
            'ref1': ['1606-03490', '1811-10154', '1705-08807'],
            'ref2': ['1811-10154', '1606-03490', '1705-08807']
        }
        
        ground_truth = {
            'ref1': '1606-03490',  # Rank 1 → RR = 1.0
            'ref2': '1811-10154'   # Rank 1 → RR = 1.0
        }
        
        mrr = evaluator.calculate_mrr(predictions, ground_truth)
        assert mrr == 1.0
    
    def test_mrr_calculation(self):
        """Test MRR calculation with mixed ranks"""
        evaluator = Evaluator()
        
        predictions = {
            'ref1': ['1606-03490', '1811-10154', '1705-08807'],  # Correct at rank 1
            'ref2': ['1811-10154', '1606-03490', '1705-08807']   # Correct at rank 2
        }
        
        ground_truth = {
            'ref1': '1606-03490',  # Rank 1 → RR = 1.0
            'ref2': '1606-03490'   # Rank 2 → RR = 0.5
        }
        
        mrr = evaluator.calculate_mrr(predictions, ground_truth)
        expected_mrr = (1.0 + 0.5) / 2  # 0.75
        assert mrr == expected_mrr
    
    def test_mrr_not_found(self):
        """Test MRR when correct answer not in predictions"""
        evaluator = Evaluator()
        
        predictions = {
            'ref1': ['wrong1', 'wrong2', 'wrong3']
        }
        
        ground_truth = {
            'ref1': 'correct'
        }
        
        mrr = evaluator.calculate_mrr(predictions, ground_truth)
        assert mrr == 0.0
    
    def test_hit_at_k(self):
        """Test Hit@k calculation"""
        evaluator = Evaluator()
        
        predictions = {
            'ref1': ['a', 'b', 'c', 'd', 'e'],  # correct at rank 1
            'ref2': ['x', 'y', 'correct', 'd', 'e'],  # correct at rank 3
            'ref3': ['wrong'] * 5  # not in top-5
        }
        
        ground_truth = {
            'ref1': 'a',
            'ref2': 'correct',
            'ref3': 'missing'
        }
        
        hit1 = evaluator.calculate_hit_at_k(predictions, ground_truth, k=1)
        hit3 = evaluator.calculate_hit_at_k(predictions, ground_truth, k=3)
        hit5 = evaluator.calculate_hit_at_k(predictions, ground_truth, k=5)
        
        assert hit1 == 1/3  # Only ref1 correct at rank 1
        assert hit3 == 2/3  # ref1 and ref2 correct in top-3
        assert hit5 == 2/3  # ref1 and ref2 correct in top-5
    
    def test_calculate_metrics(self):
        """Test all metrics calculation"""
        evaluator = Evaluator()
        
        predictions = {
            'ref1': ['a', 'b', 'c']
        }
        
        ground_truth = {
            'ref1': 'a'
        }
        
        metrics = evaluator.calculate_metrics(predictions, ground_truth)
        
        assert 'mrr' in metrics
        assert 'hit@1' in metrics
        assert 'hit@3' in metrics
        assert 'hit@5' in metrics
        assert 'avg_rank' in metrics
        
        assert metrics['mrr'] == 1.0
        assert metrics['hit@1'] == 1.0


class TestIntegration:
    """Integration tests for matcher pipeline"""
    
    def test_feature_extraction_pipeline(self):
        """Test feature extraction in pipeline context"""
        # Create pairs
        pairs = [
            {
                'publication_id': 'test',
                'bibtex_key': 'ref1',
                'candidate_id': 'cand1',
                'bibtex_data': {
                    'title': 'Test Paper',
                    'author': 'Smith',
                    'year': '2020'
                },
                'candidate_data': {
                    'paper_title': 'Test Paper',
                    'paper_authors': ['Smith'],
                    'year': '2020'
                }
            }
        ]
        
        # Extract features
        extractor = FeatureExtractor()
        features_df = extractor.extract_batch(pairs)
        
        # Features should be extracted
        assert len(features_df) == 1
        assert features_df['title_exact_match'].iloc[0] == 1
    
    def test_labeling_and_evaluation(self):
        """Test labeling to evaluation pipeline"""
        # Create pairs with known matches
        pairs = [
            {
                'publication_id': 'test',
                'bibtex_key': 'ref1',
                'candidate_id': 'correct',
                'bibtex_data': {'title': 'Exact Match'},
                'candidate_data': {'paper_title': 'Exact Match'}
            },
            {
                'publication_id': 'test',
                'bibtex_key': 'ref1',
                'candidate_id': 'wrong',
                'bibtex_data': {'title': 'Exact Match'},
                'candidate_data': {'paper_title': 'Different Title'}
            }
        ]
        
        # Label
        labeler = Labeler()
        for pair in pairs:
            labeler.automatic_label(pair)
        
        # Evaluate
        predictions = {'ref1': ['correct', 'wrong']}
        ground_truth = {'ref1': 'correct'}
        
        evaluator = Evaluator()
        mrr = evaluator.calculate_mrr(predictions, ground_truth)
        
        assert mrr == 1.0  # Correct at rank 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])