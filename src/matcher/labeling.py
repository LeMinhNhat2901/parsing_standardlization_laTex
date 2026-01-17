"""
Data Labeling Module for Reference Matching
Implements manual and automatic labeling per requirement 2.2.2

REQUIREMENTS 2.2.2:
- Manual label references for at least 5 publications (≥20 labeled pairs)
- Implement automatic matching for at least 10% of remaining non-manual data
- Clear description of label format
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from fuzzywuzzy import fuzz
import sys

sys.path.append(str(Path(__file__).parent.parent))


class Labeler:
    """
    Data labeling for reference matching
    
    LABEL FORMAT (requirement 2.2.2):
    ---------------------------------
    {
        "publication_id": {
            "bibtex_key": "arxiv_id",
            ...
        },
        ...
    }
    
    Where:
    - publication_id: arXiv ID of the paper (e.g., "2504-13946")
    - bibtex_key: Citation key from BibTeX file (e.g., "lipton2018mythos")
    - arxiv_id: Matching arXiv ID from references.json (e.g., "1606-03490")
    
    LABELING PROCESS:
    -----------------
    1. Manual labeling: Human annotator identifies correct matches
       - Required: ≥5 publications, ≥20 pairs total
       
    2. Automatic labeling: Uses heuristics (regex, string similarity)
       - Required: ≥10% of non-manually labeled data
       - Methods:
         a) Exact title match (high confidence)
         b) High title similarity (≥0.9) + same year
         c) Author overlap (≥80%) + partial title match
         d) arXiv ID extraction from BibTeX fields
    """
    
    def __init__(self):
        # Ground truth labels
        self.ground_truth = {}  # {pub_id: {bibtex_key: arxiv_id}}
        
        # Auto-labeled data
        self.auto_labels = {}  # {pub_id: {bibtex_key: arxiv_id}}
        
        # Statistics for compliance tracking
        self.statistics = {
            'manual_labeled_pubs': 0,
            'manual_labeled_pairs': 0,
            'auto_labeled_pubs': 0,
            'auto_labeled_pairs': 0,
            'total_pairs_processed': 0,
            'high_confidence_auto': 0,
            'medium_confidence_auto': 0,
        }
        
        # Confidence thresholds for auto-labeling
        self.title_exact_threshold = 1.0  # Exact match
        self.title_high_threshold = 0.90  # Very high similarity
        self.title_medium_threshold = 0.80  # Medium similarity
        self.author_overlap_threshold = 0.60  # 60% author overlap
    
    def load_manual_labels(self, labels_path: str) -> Dict:
        """
        Load manual labels from JSON file
        
        Args:
            labels_path: Path to manual_labels.json
            
        Returns:
            Dict of manual labels
        """
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.ground_truth = json.load(f)
                
            # Update statistics
            self.statistics['manual_labeled_pubs'] = len(self.ground_truth)
            self.statistics['manual_labeled_pairs'] = sum(
                len(v) for v in self.ground_truth.values()
            )
            
            return self.ground_truth
        else:
            print(f"Warning: Manual labels file not found at {labels_path}")
            return {}
    
    def save_manual_labels(self, labels_path: str):
        """Save manual labels to JSON file"""
        with open(labels_path, 'w') as f:
            json.dump(self.ground_truth, f, indent=2)
    
    def add_manual_label(self, pub_id: str, bibtex_key: str, arxiv_id: str):
        """
        Add a single manual label
        
        Args:
            pub_id: Publication ID (arXiv format)
            bibtex_key: BibTeX citation key
            arxiv_id: Matching arXiv ID from references.json
        """
        if pub_id not in self.ground_truth:
            self.ground_truth[pub_id] = {}
        
        self.ground_truth[pub_id][bibtex_key] = arxiv_id
        
        # Update statistics
        self.statistics['manual_labeled_pubs'] = len(self.ground_truth)
        self.statistics['manual_labeled_pairs'] = sum(
            len(v) for v in self.ground_truth.values()
        )
    
    def automatic_label(self, pair: Dict, use_arxiv_matching: bool = False) -> Optional[int]:
        """
        Automatically label a single pair using heuristics
        
        AUTOMATIC LABELING METHODS (per requirement 2.2.2):
        ---------------------------------------------------
        1. Regex-based: Extract arXiv ID from URL/note fields (DISABLED BY DEFAULT)
           WARNING: This can cause DATA LEAKAGE if used during training!
           The model would learn to match based on arXiv ID rather than content.
        2. String similarity: High title similarity
        3. Combined: Author + title + year matching
        
        Args:
            pair: Dict containing bibtex_data and candidate_data
            use_arxiv_matching: If True, use arXiv ID matching (ONLY for final prediction,
                               NOT for training labels to avoid data leakage!)
            
        Returns:
            1 if match, 0 if no match, None if uncertain
        """
        bibtex = pair.get('bibtex_data', {})
        candidate = pair.get('candidate_data', {})
        
        # Method 1: Extract arXiv ID from BibTeX fields
        # WARNING: DISABLED BY DEFAULT to prevent DATA LEAKAGE!
        # Using this for training labels would cause the model to achieve
        # 100% accuracy immediately because it's essentially "knowing the answer"
        # rather than learning from content features.
        if use_arxiv_matching:
            arxiv_from_bibtex = self._extract_arxiv_id(bibtex)
            candidate_id = pair.get('candidate_id', '')
            
            if arxiv_from_bibtex and candidate_id:
                # Normalize IDs for comparison
                arxiv_norm = self._normalize_arxiv_id(arxiv_from_bibtex)
                cand_norm = self._normalize_arxiv_id(candidate_id)
                
                if arxiv_norm == cand_norm:
                    return 1  # High confidence match
        
        # Method 2: Exact title match (SAFE - uses content features)
        title1 = self._normalize_text(bibtex.get('title', ''))
        title2 = self._normalize_text(
            candidate.get('paper_title', candidate.get('title', ''))
        )
        
        if title1 and title2 and len(title1) > 10 and len(title2) > 10:
            title_sim = fuzz.ratio(title1, title2) / 100.0
            
            # Exact title match = very high confidence
            if title_sim >= self.title_exact_threshold:
                return 1
            
            # Very high similarity + same year
            if title_sim >= self.title_high_threshold:
                year1 = self._extract_year(bibtex.get('year', ''))
                year2 = self._extract_year(candidate.get('year', ''))
                
                if year1 and year2 and abs(year1 - year2) <= 1:
                    return 1
                # Removed: "Missing year but high title similarity" case
                # This was too aggressive and caused false positives
            
            # Method 3: Combined author + title matching (stricter threshold)
            if title_sim >= self.title_medium_threshold:
                author_overlap = self._calculate_author_overlap(
                    bibtex.get('author', bibtex.get('authors', '')),
                    candidate.get('paper_authors', candidate.get('authors', ''))
                )
                
                # Require BOTH high title similarity AND good author overlap
                if author_overlap >= self.author_overlap_threshold:
                    return 1
        
        # Cannot confidently label - return None (will be treated as 0)
        return None
    
    def automatic_label_batch(self, pairs: List[Dict], 
                              target_percentage: float = 0.1,
                              strict: bool = False) -> int:
        """
        Automatically label a batch of pairs
        
        REQUIREMENT 2.2.2: Auto-label ≥10% of non-manual data
        
        Args:
            pairs: List of pair dicts
            target_percentage: Target percentage to label (default 0.1 = 10%)
            strict: If True, stop at target percentage
            
        Returns:
            Number of pairs auto-labeled
        """
        # Filter out pairs that already have manual labels
        unlabeled_pairs = []
        
        for pair in pairs:
            pub_id = pair.get('publication_id', '')
            bibtex_key = pair.get('bibtex_key', '')
            
            # Skip if already manually labeled
            if pub_id in self.ground_truth:
                if bibtex_key in self.ground_truth[pub_id]:
                    continue
            
            unlabeled_pairs.append(pair)
        
        self.statistics['total_pairs_processed'] = len(pairs)
        self.statistics['total_pubs_in_data'] = len(set(p.get('publication_id', '') for p in pairs))
        
        # Target number of pairs to auto-label
        target_count = int(len(unlabeled_pairs) * target_percentage)
        auto_labeled_count = 0
        
        print(f"   Auto-labeling target: {target_count} pairs ({target_percentage*100:.0f}%)")
        print(f"   Unlabeled pairs available: {len(unlabeled_pairs)}")
        
        for pair in unlabeled_pairs:
            pub_id = pair.get('publication_id', '')
            bibtex_key = pair.get('bibtex_key', '')
            candidate_id = pair.get('candidate_id', '')
            
            # Try to auto-label
            label = self.automatic_label(pair)
            
            if label == 1:
                # Store auto-label
                if pub_id not in self.auto_labels:
                    self.auto_labels[pub_id] = {}
                
                # Only store if this is the first match for this bibtex_key
                if bibtex_key not in self.auto_labels[pub_id]:
                    self.auto_labels[pub_id][bibtex_key] = candidate_id
                    auto_labeled_count += 1
                    
                    if strict and auto_labeled_count >= target_count:
                        break
        
        # Update statistics
        self.statistics['auto_labeled_pubs'] = len(self.auto_labels)
        self.statistics['auto_labeled_pairs'] = sum(
            len(v) for v in self.auto_labels.values()
        )
        
        print(f"   Auto-labeled: {auto_labeled_count} pairs")
        print(f"   Auto-labeled publications: {self.statistics['auto_labeled_pubs']}")
        
        return auto_labeled_count
    
    def get_label(self, pub_id: str, bibtex_key: str, candidate_id: str) -> int:
        """
        Get label for a specific pair
        
        Priority:
        1. Manual labels (ground truth)
        2. Auto-labels
        3. Default to 0 (no match)
        
        Args:
            pub_id: Publication ID
            bibtex_key: BibTeX key
            candidate_id: Candidate arXiv ID
            
        Returns:
            1 if match, 0 if no match
        """
        # Check manual labels first
        if pub_id in self.ground_truth:
            if bibtex_key in self.ground_truth[pub_id]:
                true_id = self.ground_truth[pub_id][bibtex_key]
                return 1 if candidate_id == true_id else 0
        
        # Check auto-labels
        if pub_id in self.auto_labels:
            if bibtex_key in self.auto_labels[pub_id]:
                auto_id = self.auto_labels[pub_id][bibtex_key]
                return 1 if candidate_id == auto_id else 0
        
        # Default: no match
        return 0
    
    def _extract_arxiv_id(self, bibtex: Dict) -> Optional[str]:
        """
        Extract arXiv ID from BibTeX entry
        
        Looks in:
        - eprint field
        - url field
        - note field
        - doi field (for arXiv DOIs)
        """
        # Check eprint field (standard arXiv field)
        eprint = bibtex.get('eprint', '')
        if eprint:
            arxiv_id = self._parse_arxiv_id(eprint)
            if arxiv_id:
                return arxiv_id
        
        # Check URL field
        url = bibtex.get('url', '')
        if url:
            arxiv_id = self._parse_arxiv_id(url)
            if arxiv_id:
                return arxiv_id
        
        # Check note field
        note = bibtex.get('note', '')
        if note:
            arxiv_id = self._parse_arxiv_id(note)
            if arxiv_id:
                return arxiv_id
        
        # Check DOI for arXiv
        doi = bibtex.get('doi', '')
        if doi and 'arxiv' in doi.lower():
            arxiv_id = self._parse_arxiv_id(doi)
            if arxiv_id:
                return arxiv_id
        
        return None
    
    def _parse_arxiv_id(self, text: str) -> Optional[str]:
        """
        Parse arXiv ID from text using regex
        
        Handles formats:
        - 1234.56789
        - arXiv:1234.56789
        - arxiv.org/abs/1234.56789
        - hep-th/9901001 (old format)
        """
        if not text:
            return None
        
        # New format: YYMM.NNNNN
        match = re.search(r'(\d{4}\.\d{4,5})', text)
        if match:
            return match.group(1)
        
        # Old format: category/YYYYNNN
        match = re.search(r'([a-z-]+/\d{7})', text.lower())
        if match:
            return match.group(1)
        
        return None
    
    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        """Normalize arXiv ID for comparison"""
        if not arxiv_id:
            return ""
        
        # Remove version suffix
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        
        # Convert YYMM.NNNNN to YYMM-NNNNN format if needed
        arxiv_id = arxiv_id.replace('.', '-')
        
        return arxiv_id.strip().lower()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
        text = re.sub(r'[{}$\\]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _extract_year(self, year_str: str) -> Optional[int]:
        """Extract year as integer"""
        if not year_str:
            return None
        
        match = re.search(r'(\d{4})', str(year_str))
        if match:
            return int(match.group(1))
        
        return None
    
    def _calculate_author_overlap(self, authors1: str, authors2: str) -> float:
        """
        Calculate overlap ratio between two author lists
        
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        if not authors1 or not authors2:
            return 0.0
        
        # Parse author names
        names1 = self._parse_author_names(authors1)
        names2 = self._parse_author_names(authors2)
        
        if not names1 or not names2:
            return 0.0
        
        # Compare last names
        last_names1 = set(n.lower() for n in names1)
        last_names2 = set(n.lower() for n in names2)
        
        intersection = last_names1.intersection(last_names2)
        min_size = min(len(last_names1), len(last_names2))
        
        if min_size == 0:
            return 0.0
        
        return len(intersection) / min_size
    
    def _parse_author_names(self, authors_str: str) -> List[str]:
        """
        Parse author names from string
        
        Handles formats:
        - "Smith, John and Doe, Jane"
        - "John Smith, Jane Doe"
        - ["John Smith", "Jane Doe"]
        """
        if not authors_str:
            return []
        
        # Handle list input - use type() instead of isinstance() to avoid recursion
        if type(authors_str).__name__ in ('list', 'tuple'):
            authors_str = ', '.join(str(a) for a in authors_str)
        
        # Split by 'and' or comma
        parts = re.split(r'\s+and\s+|,\s*', str(authors_str))
        
        names = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Extract last name (usually last word or first word before comma)
            words = part.split()
            if words:
                # Take last word as last name
                names.append(words[-1])
        
        return names
    
    def get_statistics(self) -> Dict:
        """
        Get labeling statistics for compliance checking
        
        Returns:
            Dict with statistics and compliance status
        """
        total_labeled = (
            self.statistics['manual_labeled_pairs'] + 
            self.statistics['auto_labeled_pairs']
        )
        
        total_pairs = self.statistics.get('total_pairs_processed', 1)
        
        # Calculate auto-label percentage of non-manual data
        manual_pairs = self.statistics['manual_labeled_pairs']
        non_manual_pairs = total_pairs - manual_pairs
        
        manual_pubs = self.statistics.get('manual_labeled_pubs', 0)
        total_pubs = self.statistics.get('total_pubs_in_data', 0)
        auto_labeled_pubs = self.statistics.get('auto_labeled_pubs', 0)
        
        non_manual_pubs = total_pubs - manual_pubs
        
        if non_manual_pairs > 0:
            auto_percentage = self.statistics['auto_labeled_pubs'] / non_manual_pubs
        else:
            auto_percentage = 0.0
        
        self.statistics['auto_label_percentage'] = auto_percentage
        
        # Compliance check per requirement 2.2.2
        manual_pubs_ok = manual_pubs >= 5
        manual_pairs_ok = self.statistics['manual_labeled_pairs'] >= 20
        auto_ok = auto_percentage >= 0.10  # 10%
        
        self.statistics['compliance_summary'] = {
            'manual_publications_met': manual_pubs_ok,
            'manual_pairs_met': manual_pairs_ok,
            'auto_labeling_met': auto_ok,
            'all_requirements_met': manual_pubs_ok and manual_pairs_ok and auto_ok
        }
        
        return self.statistics
    
    def print_compliance_report(self):
        """Print detailed compliance report for requirement 2.2.2"""
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print("DATA LABELING COMPLIANCE REPORT (Requirement 2.2.2)")
        print(f"{'='*60}\n")
        
        # Manual labeling
        print("MANUAL LABELING:")
        print(f"  Publications labeled: {stats['manual_labeled_pubs']} (≥5 required)")
        print(f"  Status: {'✅ PASS' if stats['compliance_summary']['manual_publications_met'] else '❌ FAIL'}")
        print()
        print(f"  Pairs labeled: {stats['manual_labeled_pairs']} (≥20 required)")
        print(f"  Status: {'✅ PASS' if stats['compliance_summary']['manual_pairs_met'] else '❌ FAIL'}")
        print()
        
        # Automatic labeling
        print("AUTOMATIC LABELING:")
        print(f"  Auto-labeled pairs: {stats['auto_labeled_pairs']}")
        print(f"  Percentage of non-manual data: {stats.get('auto_label_percentage', 0)*100:.1f}% (≥10% required)")
        print(f"  Status: {'✅ PASS' if stats['compliance_summary']['auto_labeling_met'] else '❌ FAIL'}")
        print()
        
        # Overall
        print("OVERALL COMPLIANCE:")
        if stats['compliance_summary']['all_requirements_met']:
            print("  ✅ ALL REQUIREMENTS MET")
        else:
            print("  ❌ SOME REQUIREMENTS NOT MET")
            
        print(f"{'='*60}\n")
