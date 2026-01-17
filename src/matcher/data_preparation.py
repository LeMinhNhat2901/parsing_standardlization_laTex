"""
Prepare data for machine learning
Create all possible (BibTeX, Candidate) pairs

As per requirement 2.2.1:
- Create m×n pairs for each publication (m = bibtex entries, n = candidate refs)
- Label pairs manually (≥5 pubs, ≥20 pairs) and automatically (≥10% of dataset)
- Split data into train/val/test sets
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.file_io import load_json, load_bibtex


class DataPreparation:
    """
    Prepare data for machine learning pipeline
    
    WORKFLOW:
    1. Load BibTeX entries (refs.bib) and candidate references (references.json)
    2. Create m×n pairs for each publication
    3. Aggregate all pairs across publications
    4. Split into train/val/test sets
    """
    
    def __init__(self):
        self.pairs = []
        self.publications = {}  # pub_id -> pairs
        self.statistics = {
            'total_pairs': 0,
            'total_publications': 0,
            'avg_bibtex_per_pub': 0.0,
            'avg_candidates_per_pub': 0.0,
            'imbalance_ratio': 0.0
        }
    
    def create_pairs_for_publication(self, refs_bib_path: str, references_json_path: str, 
                                     publication_id: str = None) -> List[Dict]:
        """
        Create m × n pairs for ONE publication
        
        EXPLICIT EXAMPLE:
        ----------------
        Given:
            refs.bib contains m=10 BibTeX entries:
                - lipton2018mythos
                - rudin2019stop
                - ... (8 more)
            
            references.json contains n=50 arXiv references:
                - 1606-03490
                - 1811-10154
                - ... (48 more)
        
        Then we create: m × n = 10 × 50 = 500 pairs
        
        Each pair is:
            (lipton2018mythos, 1606-03490) → label: ?
            (lipton2018mythos, 1811-10154) → label: ?
            ...
        
        Most pairs will have label = 0 (no match)
        Only m pairs will have label = 1 (correct matches)
        
        This is a HIGHLY IMBALANCED dataset:
            - Positive class: m pairs (10)
            - Negative class: m×(n-1) pairs (490)
            - Imbalance ratio: 1:49
        
        Args:
            refs_bib_path: Path to refs.bib file
            references_json_path: Path to references.json file
            publication_id: Optional publication ID (extracted from path if not provided)
        
        Returns:
            List of pair dictionaries
        """
        # Load data
        bibtex_entries = load_bibtex(refs_bib_path)  # m entries
        references = load_json(references_json_path)  # n references
        
        if not bibtex_entries or not references:
            print(f"Warning: Empty data for {refs_bib_path}")
            return []
        
        # Extract publication ID from path if not provided
        if publication_id is None:
            publication_id = self._extract_pub_id(refs_bib_path)
        
        pairs = []
        
        # Create ALL combinations (m × n)
        # IMPORTANT: Create shallow copies of data to avoid circular references
        # that can cause RecursionError with isinstance checks
        for bibtex_key, bibtex_data in bibtex_entries.items():
            # Create a safe copy of bibtex_data with only primitive values
            safe_bibtex = self._make_safe_dict(bibtex_data)
            safe_bibtex['_original_key'] = bibtex_key
            
            for arxiv_id, ref_data in references.items():
                # Create a safe copy of ref_data
                safe_ref = self._make_safe_dict(ref_data) if ref_data else {}
                safe_ref['_original_id'] = arxiv_id
                
                pair = {
                    'publication_id': publication_id,
                    'bibtex_key': bibtex_key,
                    'bibtex_data': safe_bibtex,
                    'candidate_id': arxiv_id,
                    'candidate_data': safe_ref,
                    'label': None  # Will be filled by Labeler
                }
                pairs.append(pair)
        
        expected_count = len(bibtex_entries) * len(references)
        actual_count = len(pairs)
        
        assert actual_count == expected_count, \
            f"Expected {expected_count} pairs, got {actual_count}"
        
        # Calculate imbalance ratio
        # Assume only m pairs are positive (one match per bibtex entry)
        m = len(bibtex_entries)
        n = len(references)
        if m > 0:
            imbalance_ratio = (m * (n - 1)) / m if m < n else 1.0
        else:
            imbalance_ratio = 0.0
        
        print(f"Created {actual_count} pairs for publication {publication_id}")
        print(f"  BibTeX entries (m): {m}")
        print(f"  Candidates (n): {n}")
        print(f"  Imbalance ratio: 1:{imbalance_ratio:.1f}")
        
        return pairs
    
    def _make_safe_dict(self, d: dict, _seen: set = None) -> dict:
        """
        Create a safe copy of a dictionary with only primitive values.
        This prevents RecursionError with isinstance checks in pandas.
        
        Args:
            d: Input dictionary (may have nested structures)
            _seen: Set of seen object IDs to detect circular references
            
        Returns:
            Dict with only string values (nested dicts are JSON-serialized)
        """
        # Initialize seen set on first call
        if _seen is None:
            _seen = set()
        
        # Use type() instead of isinstance() to avoid recursion issues
        if not d or type(d).__name__ != 'dict':
            return {}
        
        # Check for circular references using object ID
        obj_id = id(d)
        if obj_id in _seen:
            return {}  # Circular reference detected, return empty
        _seen.add(obj_id)
        
        safe = {}
        for k, v in d.items():
            if v is None:
                safe[k] = ''
                continue
                
            v_type = type(v).__name__
            
            # Handle primitive types
            if v_type == 'str':
                safe[k] = v
            elif v_type in ('int', 'float', 'bool'):
                safe[k] = v
            # Handle numpy types
            elif v_type in ('int64', 'float64', 'int32', 'float32', 'int16', 'float16',
                           'int8', 'uint8', 'uint16', 'uint32', 'uint64'):
                safe[k] = float(v)
            elif v_type == 'bool_':  # numpy bool
                safe[k] = bool(v)
            # Handle sequences
            elif v_type in ('list', 'tuple'):
                # Convert list to comma-separated string of primitives
                safe_items = []
                for item in v:
                    item_type = type(item).__name__
                    if item is None:
                        continue
                    elif item_type in ('str', 'int', 'float', 'bool'):
                        safe_items.append(str(item))
                    elif item_type == 'dict':
                        # Skip nested dicts in lists to avoid recursion
                        continue
                    else:
                        safe_items.append(str(item))
                safe[k] = ', '.join(safe_items)
            # Handle nested dicts
            elif v_type == 'dict':
                # Flatten nested dict or convert to string
                try:
                    import json
                    # Create safe nested dict first
                    nested_safe = self._make_safe_dict(v, _seen.copy())
                    safe[k] = json.dumps(nested_safe) if nested_safe else ''
                except Exception:
                    safe[k] = str(v)[:200]  # Truncate to prevent huge strings
            # Handle ndarray
            elif v_type == 'ndarray':
                try:
                    if v.size <= 10:
                        safe[k] = ', '.join(str(x) for x in v.flatten())
                    else:
                        safe[k] = str(v.shape)
                except Exception:
                    safe[k] = ''
            else:
                # For any other type, convert to string (truncated)
                try:
                    safe[k] = str(v)[:200]
                except Exception:
                    safe[k] = ''
        
        return safe
    
    def _extract_pub_id(self, path: str) -> str:
        """
        Extract publication ID from file path
        
        Example:
            input: /data/2304.12345/refs.bib
            output: 2304.12345
        """
        path_obj = Path(path)
        # Publication ID is usually the parent directory name
        return path_obj.parent.name
    
    def create_all_pairs(self, publications_dir: str, 
                         refs_filename: str = 'refs.bib',
                         references_filename: str = 'references.json') -> List[Dict]:
        """
        Create pairs for all publications in directory
        
        Args:
            publications_dir: Root directory containing publication folders
            refs_filename: Name of BibTeX file in each publication folder
            references_filename: Name of references JSON file
        
        Returns:
            List of all pairs across all publications
        """
        publications_path = Path(publications_dir)
        all_pairs = []
        pub_count = 0
        total_bibtex = 0
        total_candidates = 0
        
        # Find all publication directories
        for pub_dir in publications_path.iterdir():
            if not pub_dir.is_dir():
                continue
            
            refs_path = pub_dir / refs_filename
            references_path = pub_dir / references_filename
            
            # Skip if required files don't exist
            if not refs_path.exists() or not references_path.exists():
                print(f"Skipping {pub_dir.name}: missing required files")
                continue
            
            pub_id = pub_dir.name
            pairs = self.create_pairs_for_publication(
                str(refs_path), 
                str(references_path),
                pub_id
            )
            
            if pairs:
                all_pairs.extend(pairs)
                self.publications[pub_id] = pairs
                pub_count += 1
                
                # Track statistics
                bibtex_count = len(set(p['bibtex_key'] for p in pairs))
                total_bibtex += bibtex_count
                total_candidates += len(pairs) // bibtex_count if bibtex_count > 0 else 0
        
        # Update statistics
        self.pairs = all_pairs
        self.statistics['total_pairs'] = len(all_pairs)
        self.statistics['total_publications'] = pub_count
        self.statistics['avg_bibtex_per_pub'] = total_bibtex / pub_count if pub_count > 0 else 0
        self.statistics['avg_candidates_per_pub'] = total_candidates / pub_count if pub_count > 0 else 0
        
        print(f"\nTotal statistics:")
        print(f"  Publications processed: {pub_count}")
        print(f"  Total pairs created: {len(all_pairs)}")
        
        return all_pairs
    
    def split_data(self, pairs: List[Dict], 
                   manual_pubs: List[str] = None, 
                   auto_pubs: List[str] = None,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   random_seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split into train/val/test sets
        
        Splitting strategy:
        1. Manual labeled publications → use for validation/test (high quality)
        2. Auto labeled publications → use for training (larger quantity)
        3. If not specified, split randomly by publication
        
        Args:
            pairs: All pairs
            manual_pubs: List of manually labeled publication IDs
            auto_pubs: List of automatically labeled publication IDs
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            train_pairs, val_pairs, test_pairs
        """
        random.seed(random_seed)
        
        # Group pairs by publication
        pairs_by_pub = {}
        for pair in pairs:
            pub_id = pair['publication_id']
            if pub_id not in pairs_by_pub:
                pairs_by_pub[pub_id] = []
            pairs_by_pub[pub_id].append(pair)
        
        all_pub_ids = list(pairs_by_pub.keys())
        
        if manual_pubs and auto_pubs:
            # Use specified splits
            # Manual labeled → test & validation (higher quality)
            # Auto labeled → training (larger quantity)
            
            train_pubs = auto_pubs
            test_pubs = []
            val_pubs = []
            
            # Split manual pubs between validation and test
            random.shuffle(manual_pubs)
            val_count = len(manual_pubs) // 2
            val_pubs = manual_pubs[:val_count]
            test_pubs = manual_pubs[val_count:]
            
        else:
            # Random split by publication
            random.shuffle(all_pub_ids)
            
            n = len(all_pub_ids)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_pubs = all_pub_ids[:train_end]
            val_pubs = all_pub_ids[train_end:val_end]
            test_pubs = all_pub_ids[val_end:]
        
        # Collect pairs for each split
        train_pairs = []
        val_pairs = []
        test_pairs = []
        
        for pub_id, pub_pairs in pairs_by_pub.items():
            if pub_id in train_pubs:
                train_pairs.extend(pub_pairs)
            elif pub_id in val_pubs:
                val_pairs.extend(pub_pairs)
            elif pub_id in test_pubs:
                test_pairs.extend(pub_pairs)
        
        print(f"\nData split:")
        print(f"  Train: {len(train_pairs)} pairs ({len(train_pubs)} pubs)")
        print(f"  Val:   {len(val_pairs)} pairs ({len(val_pubs)} pubs)")
        print(f"  Test:  {len(test_pairs)} pairs ({len(test_pubs)} pubs)")
        
        return train_pairs, val_pairs, test_pairs
    
    def get_statistics(self) -> Dict:
        """Return data statistics"""
        return self.statistics.copy()
    
    def get_pairs_for_publication(self, pub_id: str) -> List[Dict]:
        """Get all pairs for a specific publication"""
        return self.publications.get(pub_id, [])
    
    def filter_labeled_pairs(self, pairs: List[Dict]) -> List[Dict]:
        """Filter to only include pairs with labels"""
        return [p for p in pairs if p.get('label') is not None]
    
    def get_positive_pairs(self, pairs: List[Dict]) -> List[Dict]:
        """Get all positive pairs (label = 1)"""
        return [p for p in pairs if p.get('label') == 1]
    
    def get_negative_pairs(self, pairs: List[Dict]) -> List[Dict]:
        """Get all negative pairs (label = 0)"""
        return [p for p in pairs if p.get('label') == 0]
    
    def balance_pairs(self, pairs: List[Dict], 
                      strategy: str = 'undersample',
                      ratio: float = 1.0,
                      random_seed: int = 42) -> List[Dict]:
        """
        Balance positive/negative pairs
        
        Args:
            pairs: Input pairs
            strategy: 'undersample' or 'oversample'
            ratio: Negative/positive ratio (1.0 = equal)
            random_seed: Random seed
            
        Returns:
            Balanced pairs
        """
        random.seed(random_seed)
        
        positive = self.get_positive_pairs(pairs)
        negative = self.get_negative_pairs(pairs)
        
        if not positive:
            return pairs
        
        target_negative = int(len(positive) * ratio)
        
        if strategy == 'undersample':
            # Reduce negative samples
            if len(negative) > target_negative:
                negative = random.sample(negative, target_negative)
        
        elif strategy == 'oversample':
            # Increase positive samples
            target_positive = len(negative) // ratio
            while len(positive) < target_positive:
                positive.extend(random.sample(positive, 
                               min(len(positive), target_positive - len(positive))))
        
        balanced = positive + negative
        random.shuffle(balanced)
        
        print(f"Balanced pairs: {len(positive)} positive, {len(negative)} negative")
        
        return balanced