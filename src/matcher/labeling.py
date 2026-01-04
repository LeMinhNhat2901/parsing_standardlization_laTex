"""
Label (BibTeX, Candidate) pairs as match/no-match

As per requirement 2.2.1:
- Manually label ≥5 publications, ≥20 BibTeX-arXiv pairs
- Automatically label ≥10% of dataset using heuristics
- Store ground truth for evaluation
"""

from fuzzywuzzy import fuzz
import re
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class Labeler:
    """
    Label pairs for supervised learning
    
    Supports:
    1. Manual labeling (from pre-labeled files)
    2. Automatic labeling (heuristics-based)
    3. Ground truth management
    """
    
    def __init__(self, 
                 title_threshold: float = 90,
                 author_threshold: float = 70,
                 year_tolerance: int = 1):
        """
        Args:
            title_threshold: Minimum title similarity for auto-labeling
            author_threshold: Minimum author overlap for auto-labeling
            year_tolerance: Maximum year difference allowed
        """
        self.title_threshold = title_threshold
        self.author_threshold = author_threshold
        self.year_tolerance = year_tolerance
        
        self.labels = {}  # (pub_id, bibtex_key, candidate_id) -> label
        self.ground_truth = {}  # pub_id -> {bibtex_key: correct_arxiv_id}
        self.manual_labels = {}  # Separate tracking for manual labels
        self.auto_labels = {}  # Separate tracking for auto labels
        
        self.statistics = {
            'manual_labeled_pubs': 0,
            'manual_labeled_pairs': 0,
            'auto_labeled_pairs': 0,
            'total_positive': 0,
            'total_negative': 0,
            'uncertain': 0
        }
    
    def manual_label(self, pair: Dict, label: int) -> int:
        """
        Manually label a pair
        
        Args:
            pair: Pair dict containing bibtex_key, candidate_id, etc.
            label: 1 for match, 0 for no match
            
        Returns:
            The assigned label
        """
        key = (pair['publication_id'], pair['bibtex_key'], pair['candidate_id'])
        self.labels[key] = label
        self.manual_labels[key] = label
        pair['label'] = label
        
        if label == 1:
            self.statistics['total_positive'] += 1
        else:
            self.statistics['total_negative'] += 1
        
        return label
    
    def load_manual_labels(self, labels_path: str):
        """
        Load pre-labeled data from JSON file
        
        Expected format:
        {
            "publication_id": {
                "bibtex_key": "correct_arxiv_id",
                ...
            },
            ...
        }
        
        Args:
            labels_path: Path to labels JSON file
        """
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        
        for pub_id, mappings in labels_data.items():
            self.ground_truth[pub_id] = mappings
            self.statistics['manual_labeled_pubs'] += 1
            
            for bibtex_key, arxiv_id in mappings.items():
                self.statistics['manual_labeled_pairs'] += 1
        
        print(f"Loaded {self.statistics['manual_labeled_pubs']} publications with "
              f"{self.statistics['manual_labeled_pairs']} labeled pairs")
    
    def automatic_label(self, pair: Dict, strict: bool = False) -> Optional[int]:
        """
        Automatically label using heuristics
        
        Heuristics:
        - Title exact match → 1
        - Title high similarity (>90%) + author overlap → 1
        - Year match + first author match → likely 1
        - Low title similarity (<50%) → 0
        - Otherwise → None (uncertain)
        
        Args:
            pair: Pair dict
            strict: If True, only label high-confidence pairs
            
        Returns:
            1 if match, 0 if no match, None if uncertain
        """
        bibtex = pair.get('bibtex_data', {})
        candidate = pair.get('candidate_data', {})
        
        # Extract fields
        title1 = bibtex.get('title', '')
        title2 = candidate.get('paper_title', candidate.get('title', ''))
        
        authors1 = bibtex.get('author', bibtex.get('authors', ''))
        authors2 = candidate.get('paper_authors', candidate.get('authors', ''))
        
        year1 = str(bibtex.get('year', ''))
        year2 = str(candidate.get('year', ''))
        
        # Calculate similarities
        title_sim = self._title_similarity(title1, title2)
        author_overlap = self._author_overlap(authors1, authors2)
        year_match = self._year_match(year1, year2)
        
        label = None
        confidence = 'low'
        
        # Rule 1: Exact title match → definite positive
        if title_sim >= 99:
            label = 1
            confidence = 'high'
        
        # Rule 2: High title similarity + author overlap → positive
        elif title_sim >= self.title_threshold and author_overlap >= self.author_threshold:
            label = 1
            confidence = 'high'
        
        # Rule 3: Title similar + year match + first author match
        elif title_sim >= 80 and year_match and author_overlap >= 50:
            label = 1
            confidence = 'medium'
        
        # Rule 4: Very low title similarity → definite negative
        elif title_sim < 30:
            label = 0
            confidence = 'high'
        
        # Rule 5: Low title similarity + no author overlap → negative
        elif title_sim < 50 and author_overlap < 30:
            label = 0
            confidence = 'medium'
        
        # Rule 6: Moderate similarity but wrong year → likely negative
        elif title_sim < 70 and not year_match:
            label = 0
            confidence = 'low'
        
        # If strict mode, only accept high confidence labels
        if strict and confidence != 'high':
            label = None
        
        # Track and apply label
        if label is not None:
            key = (pair['publication_id'], pair['bibtex_key'], pair['candidate_id'])
            self.labels[key] = label
            self.auto_labels[key] = label
            pair['label'] = label
            
            self.statistics['auto_labeled_pairs'] += 1
            if label == 1:
                self.statistics['total_positive'] += 1
            else:
                self.statistics['total_negative'] += 1
        else:
            self.statistics['uncertain'] += 1
        
        return label
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate title similarity (0-100)
        
        Uses multiple metrics and takes the best one
        """
        if not title1 or not title2:
            return 0.0
        
        # Normalize
        t1 = self._normalize_title(title1)
        t2 = self._normalize_title(title2)
        
        if not t1 or not t2:
            return 0.0
        
        # Calculate multiple similarity metrics
        ratio = fuzz.ratio(t1, t2)
        partial = fuzz.partial_ratio(t1, t2)
        token_sort = fuzz.token_sort_ratio(t1, t2)
        token_set = fuzz.token_set_ratio(t1, t2)
        
        # Return the highest (most generous) match
        return max(ratio, partial, token_sort, token_set)
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison"""
        if not title:
            return ""
        
        # Convert to lowercase
        title = title.lower()
        
        # Remove LaTeX commands
        title = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', title)
        title = re.sub(r'[{}$\\]', '', title)
        
        # Remove punctuation
        title = re.sub(r'[^\w\s]', ' ', title)
        
        # Normalize whitespace
        title = ' '.join(title.split())
        
        return title.strip()
    
    def _author_overlap(self, authors1, authors2) -> float:
        """
        Calculate author overlap ratio (0-100)
        
        Compares by last names
        """
        names1 = self._extract_last_names(authors1)
        names2 = self._extract_last_names(authors2)
        
        if not names1 or not names2:
            return 0.0
        
        # Calculate overlap
        intersection = names1.intersection(names2)
        min_size = min(len(names1), len(names2))
        
        return (len(intersection) / min_size) * 100 if min_size > 0 else 0.0
    
    def _extract_last_names(self, authors) -> set:
        """Extract last names from author string/list"""
        if not authors:
            return set()
        
        # Convert to string if list
        if isinstance(authors, list):
            authors = ' and '.join(authors)
        
        authors = str(authors)
        
        # Split by 'and' or comma
        parts = re.split(r'\s+and\s+|,\s*', authors)
        
        last_names = set()
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Remove et al.
            if 'et al' in part.lower():
                part = part.split()[0]
            
            # Get last name (assuming "First Last" or "Last, First" format)
            words = part.split()
            if words:
                # Check for "Last, First" format
                if ',' in part:
                    last_name = words[0].rstrip(',')
                else:
                    last_name = words[-1]
                
                # Normalize
                last_name = re.sub(r'[^\w]', '', last_name.lower())
                if last_name and len(last_name) > 1:
                    last_names.add(last_name)
        
        return last_names
    
    def _year_match(self, year1: str, year2: str) -> bool:
        """Check if years match within tolerance"""
        try:
            # Extract 4-digit years
            y1_match = re.search(r'\d{4}', str(year1))
            y2_match = re.search(r'\d{4}', str(year2))
            
            if not y1_match or not y2_match:
                return False
            
            y1 = int(y1_match.group())
            y2 = int(y2_match.group())
            
            return abs(y1 - y2) <= self.year_tolerance
        except:
            return False
    
    def label_publication(self, publication_id: str, pairs: List[Dict], 
                          method: str = 'auto') -> Dict:
        """
        Label all pairs for one publication
        
        Args:
            publication_id: Publication ID
            pairs: List of pairs for this publication
            method: 'manual' or 'auto'
            
        Returns:
            Dict: {(bibtex_key, candidate_id): label}
        """
        results = {}
        
        # First, apply ground truth if available
        if publication_id in self.ground_truth:
            gt = self.ground_truth[publication_id]
            
            for pair in pairs:
                bibtex_key = pair['bibtex_key']
                candidate_id = pair['candidate_id']
                
                if bibtex_key in gt:
                    # This bibtex entry has a known match
                    if gt[bibtex_key] == candidate_id:
                        label = 1  # This is the correct match
                    else:
                        label = 0  # This is not the correct match
                    
                    self.manual_label(pair, label)
                    results[(bibtex_key, candidate_id)] = label
                    
                elif method == 'auto':
                    # No ground truth, use automatic labeling
                    label = self.automatic_label(pair)
                    if label is not None:
                        results[(bibtex_key, candidate_id)] = label
        
        elif method == 'auto':
            # No ground truth available, use automatic labeling
            for pair in pairs:
                label = self.automatic_label(pair)
                if label is not None:
                    bibtex_key = pair['bibtex_key']
                    candidate_id = pair['candidate_id']
                    results[(bibtex_key, candidate_id)] = label
        
        return results
    
    def get_ground_truth(self, publication_id: str) -> Dict[str, str]:
        """
        Get ground truth for pred.json evaluation
        
        Returns:
            Dict: {bibtex_key: correct_arxiv_id}
        """
        return self.ground_truth.get(publication_id, {})
    
    def get_all_ground_truth(self) -> Dict:
        """Get all ground truth data"""
        return self.ground_truth.copy()
    
    def set_ground_truth(self, publication_id: str, bibtex_key: str, arxiv_id: str):
        """
        Set ground truth for a specific pair
        
        Args:
            publication_id: Publication ID
            bibtex_key: BibTeX entry key
            arxiv_id: Correct arXiv ID
        """
        if publication_id not in self.ground_truth:
            self.ground_truth[publication_id] = {}
        
        self.ground_truth[publication_id][bibtex_key] = arxiv_id
    
    def save_ground_truth(self, output_path: str):
        """Save ground truth to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.ground_truth, f, indent=2)
        
        print(f"Saved ground truth to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Return labeling statistics"""
        return self.statistics.copy()
    
    def get_label(self, pair: Dict) -> Optional[int]:
        """Get label for a pair"""
        key = (pair['publication_id'], pair['bibtex_key'], pair['candidate_id'])
        return self.labels.get(key)
    
    def is_manually_labeled(self, pair: Dict) -> bool:
        """Check if pair was manually labeled"""
        key = (pair['publication_id'], pair['bibtex_key'], pair['candidate_id'])
        return key in self.manual_labels
    
    def clear_labels(self):
        """Clear all labels"""
        self.labels.clear()
        self.manual_labels.clear()
        self.auto_labels.clear()
        self.statistics = {
            'manual_labeled_pubs': 0,
            'manual_labeled_pairs': 0,
            'auto_labeled_pairs': 0,
            'total_positive': 0,
            'total_negative': 0,
            'uncertain': 0
        }