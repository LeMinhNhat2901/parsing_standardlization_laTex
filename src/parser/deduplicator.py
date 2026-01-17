"""
Deduplication for:
1. Reference entries (with \\cite{} renaming)
2. Full-text content across versions

As per requirement 2.1.3:
- Reference Entries: Deduplicate across versions, choose single citation key,
  rename \\cite{} commands, unionize fields of duplicate entries
- Full-text Content: If element's text matches exactly across versions,
  represent by single identifier (after cleanup)
"""

import re
import sys
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path

# CRITICAL: Increase recursion limit before any processing
if sys.getrecursionlimit() < 10000:
    sys.setrecursionlimit(10000)

from fuzzywuzzy import fuzz

sys.path.append(str(Path(__file__).parent.parent))


class Deduplicator:
    """
    Handle deduplication for references and content
    """
    
    def __init__(self, title_threshold=95, author_threshold=80):
        """
        Args:
            title_threshold: Similarity threshold for title matching (0-100)
            author_threshold: Overlap threshold for author matching (0-100)
        """
        self.title_threshold = title_threshold
        self.author_threshold = author_threshold
        self.rename_map = {}  # old_key -> new_key
    
    def deduplicate_references(self, bibtex_entries: Dict, latex_files: List[str] = None) -> Dict:
        """
        Deduplicate BibTeX entries and rename \\cite{} commands
        
        Steps:
        1. Find duplicate references (by title/author similarity)
        2. Choose canonical key for each duplicate group
        3. Create rename map
        4. Rename all \\cite{} commands in LaTeX files
        5. Unionize fields of duplicate entries
        
        Args:
            bibtex_entries: Dict of BibTeX entries {key: entry_dict}
            latex_files: List of LaTeX file paths to update (optional)
        
        Returns:
            Deduplicated BibTeX entries
        """
        if not bibtex_entries:
            return {}
        
        # Step 1: Find duplicates
        duplicate_groups = self._find_duplicates(bibtex_entries)
        
        # Step 2 & 3: Choose canonical key and create rename map
        self.rename_map = {}
        deduplicated = {}
        processed_keys = set()
        
        for group in duplicate_groups:
            if len(group) == 1:
                # No duplicates
                key = group[0]
                deduplicated[key] = bibtex_entries[key]
                processed_keys.add(key)
            else:
                # Has duplicates - choose canonical key
                canonical_key = self._choose_canonical_key(group, bibtex_entries)
                
                # Create rename map for all other keys
                for key in group:
                    if key != canonical_key:
                        self.rename_map[key] = canonical_key
                    processed_keys.add(key)
                
                # Unionize fields
                merged_entry = self._unionize_fields(group, bibtex_entries)
                merged_entry['ID'] = canonical_key
                deduplicated[canonical_key] = merged_entry
        
        # Add any remaining entries not in groups
        for key in bibtex_entries:
            if key not in processed_keys:
                deduplicated[key] = bibtex_entries[key]
        
        # Step 4: Rename citations in LaTeX files
        if latex_files and self.rename_map:
            self._rename_citations_in_files(latex_files, self.rename_map)
        
        return deduplicated
    
    def _find_duplicates(self, entries: Dict) -> List[List[str]]:
        """
        Find duplicate reference entries
        Compare by: title similarity + author overlap
        
        Returns:
            List of duplicate groups [[key1, key2], [key3, key4, key5], ...]
        """
        keys = list(entries.keys())
        visited = set()
        groups = []
        
        for i, key1 in enumerate(keys):
            if key1 in visited:
                continue
            
            group = [key1]
            visited.add(key1)
            
            entry1 = entries[key1]
            
            for j in range(i + 1, len(keys)):
                key2 = keys[j]
                if key2 in visited:
                    continue
                
                entry2 = entries[key2]
                
                if self._is_duplicate(entry1, entry2):
                    group.append(key2)
                    visited.add(key2)
            
            groups.append(group)
        
        return groups
    
    def _is_duplicate(self, entry1: Dict, entry2: Dict) -> bool:
        """Check if two entries are duplicates"""
        # Get titles
        title1 = entry1.get('title', '').lower().strip()
        title2 = entry2.get('title', '').lower().strip()
        
        if not title1 or not title2:
            return False
        
        # Calculate title similarity
        title_sim = fuzz.ratio(title1, title2)
        
        if title_sim >= self.title_threshold:
            return True
        
        # If title is somewhat similar, also check authors
        if title_sim >= 70:
            # Get authors
            authors1 = entry1.get('author', '')
            authors2 = entry2.get('author', '')
            
            if authors1 and authors2:
                author_sim = self._calculate_author_overlap(authors1, authors2)
                
                # Both title and author match
                if title_sim >= 80 and author_sim >= self.author_threshold:
                    return True
        
        return False
    
    def _calculate_author_overlap(self, authors1, authors2) -> float:
        """Calculate author overlap percentage"""
        # Parse author strings
        def parse_authors(auth_str):
            # Use type() instead of isinstance() to avoid recursion issues
            if type(auth_str).__name__ == 'list':
                return set(a.lower().strip() for a in auth_str)
            # Split by 'and' or comma
            authors = re.split(r'\s+and\s+|,\s*', str(auth_str))
            return set(a.lower().strip() for a in authors if a.strip())
        
        set1 = parse_authors(authors1)
        set2 = parse_authors(authors2)
        
        if not set1 or not set2:
            return 0.0
        
        # Extract last names for comparison
        def get_last_names(author_set):
            last_names = set()
            for author in author_set:
                parts = author.split()
                if parts:
                    last_names.add(parts[-1])
            return last_names
        
        last1 = get_last_names(set1)
        last2 = get_last_names(set2)
        
        if not last1 or not last2:
            return 0.0
        
        intersection = last1.intersection(last2)
        min_size = min(len(last1), len(last2))
        
        return (len(intersection) / min_size) * 100
    
    def _choose_canonical_key(self, duplicate_group: List[str], entries: Dict) -> str:
        """
        Choose which key to keep as canonical
        
        Heuristics:
        - Prefer keys with more complete information
        - Prefer shorter keys (more likely to be clean)
        - Prefer first author's last name + year format
        
        Returns:
            Chosen canonical key
        """
        scored_keys = []
        
        for key in duplicate_group:
            score = 0
            entry = entries[key]
            
            # Score by completeness of entry
            if entry.get('title'):
                score += 10
            if entry.get('author'):
                score += 10
            if entry.get('year'):
                score += 5
            if entry.get('journal') or entry.get('booktitle'):
                score += 3
            
            # Prefer standard key format (name + year)
            if re.match(r'^[a-z]+\d{4}', key.lower()):
                score += 5
            
            # Penalize very long keys
            if len(key) > 30:
                score -= 3
            
            scored_keys.append((key, score, len(key)))
        
        # Sort by score (desc), then by length (asc)
        scored_keys.sort(key=lambda x: (-x[1], x[2]))
        
        return scored_keys[0][0]
    
    def _rename_citations_in_files(self, latex_files: List[str], rename_map: Dict):
        """
        Rename all \\cite{old_key} to \\cite{new_key} in files
        
        Handles:
        - \\cite{key1}
        - \\cite{key1,key2,key3}
        - \\cite[page]{key1}
        - \\citep{key1}, \\citet{key1}, etc.
        """
        for file_path in latex_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                modified = False
                
                # Pattern to match cite commands
                cite_pattern = re.compile(
                    r'(\\cite[pt]?\*?(?:\[[^\]]*\])?)\{([^}]+)\}'
                )
                
                def replace_keys(match):
                    nonlocal modified
                    prefix = match.group(1)
                    keys_str = match.group(2)
                    
                    # Split keys
                    keys = [k.strip() for k in keys_str.split(',')]
                    
                    # Replace keys
                    new_keys = []
                    for key in keys:
                        if key in rename_map:
                            new_keys.append(rename_map[key])
                            modified = True
                        else:
                            new_keys.append(key)
                    
                    return f"{prefix}{{{','.join(new_keys)}}}"
                
                new_content = cite_pattern.sub(replace_keys, content)
                
                if modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
            
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
    
    def _unionize_fields(self, duplicate_group: List[str], entries: Dict) -> Dict:
        """
        Merge fields from duplicate entries
        Union ALL fields, prefer non-empty values
        
        Example:
            Entry 1: {title: "...", year: 2018, venue: ""}
            Entry 2: {title: "...", year: 2018, venue: "ICSE"}
            
            Result: {title: "...", year: 2018, venue: "ICSE"}
        """
        merged = {}
        all_fields = set()
        
        # Collect all possible fields
        for key in duplicate_group:
            all_fields.update(entries[key].keys())
        
        # For each field, collect all non-empty values
        for field in all_fields:
            values = []
            for key in duplicate_group:
                val = entries[key].get(field)
                # Use type() instead of isinstance() to avoid recursion issues
                if val and (type(val).__name__ == 'str' and val.strip() or 
                           type(val).__name__ == 'list' and len(val) > 0):
                    values.append(val)
            
            if not values:
                continue
            
            # Union strategy depends on field type
            if field in ['author', 'authors']:
                # For authors: union all unique authors
                all_authors = []
                for v in values:
                    # Use type() instead of isinstance() to avoid recursion issues
                    if type(v).__name__ == 'list':
                        all_authors.extend(v)
                    else:
                        # Split by 'and'
                        parts = re.split(r'\s+and\s+', str(v))
                        all_authors.extend(parts)
                # Remove duplicates while preserving order
                seen = set()
                unique_authors = []
                for a in all_authors:
                    a_clean = a.strip().lower()
                    if a_clean and a_clean not in seen:
                        seen.add(a_clean)
                        unique_authors.append(a.strip())
                merged[field] = ' and '.join(unique_authors)
            
            elif field in ['title']:
                # For title: prefer longest/most complete
                merged[field] = max(values, key=len)
            
            elif field in ['year']:
                # For year: prefer first valid year
                for v in values:
                    if re.search(r'\d{4}', str(v)):
                        merged[field] = v
                        break
            
            elif field in ['journal', 'booktitle', 'venue']:
                # For venue: prefer non-empty
                merged[field] = values[0]
            
            elif field == 'ENTRYTYPE':
                # For ENTRYTYPE: prefer more specific types over 'misc'
                type_priority = {'article': 3, 'inproceedings': 3, 'book': 2, 'incollection': 2, 'misc': 1}
                best_type = 'misc'
                best_priority = 0
                for v in values:
                    priority = type_priority.get(str(v).lower(), 1)
                    if priority > best_priority:
                        best_priority = priority
                        best_type = v
                merged[field] = best_type
            
            elif field not in ['ID']:
                # For other fields: keep first non-empty
                merged[field] = values[0]
        
        # CRITICAL: Ensure ENTRYTYPE is always present
        if 'ENTRYTYPE' not in merged:
            merged['ENTRYTYPE'] = 'misc'
        
        return merged
    
    def deduplicate_content(self, elements_dict: Dict) -> Dict:
        """
        Deduplicate identical content across versions
        If text matches exactly after cleanup → use same ID
        
        Args:
            elements_dict: Dict of {id: content}
        
        Returns:
            Deduplicated elements dict with mapping
        """
        # Build content hash -> id mapping
        content_to_id = {}
        id_mapping = {}  # old_id -> new_id
        deduplicated = {}
        
        for elem_id, content in elements_dict.items():
            # Normalize content for comparison
            # Use type() instead of isinstance() to avoid recursion issues
            if type(content).__name__ == 'dict':
                # For figures/tables, use caption
                normalized = self._normalize_for_comparison(content.get('caption', ''))
            else:
                normalized = self._normalize_for_comparison(str(content))
            
            if normalized in content_to_id:
                # Duplicate found
                canonical_id = content_to_id[normalized]
                id_mapping[elem_id] = canonical_id
            else:
                # New unique content
                content_to_id[normalized] = elem_id
                deduplicated[elem_id] = content
                id_mapping[elem_id] = elem_id
        
        return deduplicated, id_mapping
    
    def _normalize_for_comparison(self, text: str) -> str:
        """
        Normalize text for comparison
        - Remove extra whitespace
        - Lowercase
        - Remove punctuation variations
        """
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize quotes and dashes
        text = re.sub(r'[""\'\'`]', '"', text)
        text = re.sub(r'[–—−]', '-', text)
        
        # Remove some punctuation variations
        text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)
        
        return text.strip()
    
    def get_rename_map(self) -> Dict[str, str]:
        """Get the citation rename map"""
        return self.rename_map.copy()
    
    def get_statistics(self) -> Dict:
        """Get deduplication statistics"""
        return {
            'citations_renamed': len(self.rename_map),
            'rename_map': self.rename_map
        }