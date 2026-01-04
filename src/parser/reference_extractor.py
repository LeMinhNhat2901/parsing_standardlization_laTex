"""
Extract and convert references to BibTeX format
Handles \\bibitem and existing .bib files

As per requirement 2.1.3:
- For citations defined using \\bibitem within .tex files, convert them 
  into standard BibTeX entries using programmatic tools (e.g., Regular Expressions)
"""

import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
import re
from pathlib import Path
from typing import Dict, List, Optional


class ReferenceExtractor:
    """
    Extract references from LaTeX files and convert to BibTeX format
    """
    
    def __init__(self):
        self.bibtex_entries = {}
        
        # Patterns for extracting information from bibitem
        self.author_patterns = [
            # "Author, A. and Author, B."
            re.compile(r'^([A-Z][a-zéèêëàâäùûüîïôö\'-]+(?:\s+[A-Z]\.?)*(?:\s+and\s+[A-Z][a-zéèêëàâäùûüîïôö\'-]+(?:\s+[A-Z]\.?)*)*)', re.IGNORECASE),
            # "A. Author, B. Author"
            re.compile(r'^([A-Z]\.?\s+[A-Z][a-z]+(?:,?\s+(?:and\s+)?[A-Z]\.?\s+[A-Z][a-z]+)*)', re.IGNORECASE),
        ]
        
        self.year_patterns = [
            re.compile(r'\((\d{4})\)'),      # (2020)
            re.compile(r',\s*(\d{4})\.'),    # , 2020.
            re.compile(r'(\d{4})\.'),        # 2020.
            re.compile(r'\b(\d{4})\b'),      # any 4-digit year
        ]
        
        self.title_patterns = [
            # "Title" or ``Title''
            re.compile(r'[""``]([^"``\'\']+)[""\'\']+'),
            # Title after author. - look for sentence ending with period before venue
            re.compile(r'\.?\s+([^.]+?)\.\s+(?:In\s+|Proceedings|Journal|Conference|arXiv)', re.IGNORECASE),
        ]
        
        self.venue_patterns = [
            re.compile(r'In\s+(?:Proceedings\s+of\s+(?:the\s+)?)?([^.]+?)(?:,|\d{4}|\.)', re.IGNORECASE),
            re.compile(r'(?:Journal|Transactions)\s+(?:of|on)\s+([^.]+?)(?:,|\d{4}|\.)', re.IGNORECASE),
            re.compile(r'arXiv(?:\s+preprint)?[:\s]*([^\s,]+)', re.IGNORECASE),
        ]
    
    def extract_from_bibitem(self, latex_content: str) -> Dict:
        """
        Extract \\bibitem entries and convert to BibTeX
        
        Pattern:
            \\bibitem{key}
            Author et al. Title. Journal, year.
        
        Args:
            latex_content: LaTeX content containing bibliography
        
        Returns:
            Dict of BibTeX entries {key: entry_dict}
        """
        entries = {}
        
        # Pattern to match \bibitem{key} and capture following text
        bibitem_pattern = re.compile(
            r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}\s*\n?(.*?)(?=\\bibitem|\\end\{thebibliography\}|$)',
            re.DOTALL
        )
        
        for match in bibitem_pattern.finditer(latex_content):
            key = match.group(1).strip()
            bibitem_text = match.group(2).strip()
            
            # Convert to BibTeX entry
            entry = self.bibitem_to_bibtex(key, bibitem_text)
            entries[key] = entry
        
        self.bibtex_entries.update(entries)
        return entries
    
    def extract_from_bib_file(self, bib_file_path) -> Dict:
        """
        Parse existing .bib file
        
        Args:
            bib_file_path: Path to .bib file
        
        Returns:
            Dict of BibTeX entries {key: entry_dict}
        """
        bib_file_path = Path(bib_file_path)
        
        if not bib_file_path.exists():
            return {}
        
        with open(bib_file_path, 'r', encoding='utf-8') as f:
            bib_database = bibtexparser.load(f)
        
        entries = {}
        for entry in bib_database.entries:
            key = entry.get('ID', '')
            if key:
                entries[key] = entry
        
        self.bibtex_entries.update(entries)
        return entries
    
    def bibitem_to_bibtex(self, key: str, bibitem_text: str) -> Dict:
        """
        Convert a single \\bibitem to BibTeX format
        Uses regex to extract: authors, title, year, venue
        
        Args:
            key: Citation key
            bibitem_text: Text content after \\bibitem{key}
        
        Returns:
            BibTeX entry dict
        """
        # Clean up the text
        text = bibitem_text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        entry = {
            'ID': key,
            'ENTRYTYPE': 'misc',  # Default type
        }
        
        # Extract authors
        authors = self._extract_authors(text)
        if authors:
            entry['author'] = authors
        
        # Extract year
        year = self._extract_year(text)
        if year:
            entry['year'] = year
        
        # Extract title
        title = self._extract_title(text)
        if title:
            entry['title'] = title
        
        # Extract venue
        venue = self._extract_venue(text)
        if venue:
            # Determine entry type based on venue
            if 'conference' in venue.lower() or 'proceedings' in venue.lower():
                entry['ENTRYTYPE'] = 'inproceedings'
                entry['booktitle'] = venue
            elif 'journal' in venue.lower() or 'transactions' in venue.lower():
                entry['ENTRYTYPE'] = 'article'
                entry['journal'] = venue
            elif 'arxiv' in venue.lower():
                entry['ENTRYTYPE'] = 'misc'
                entry['eprint'] = venue
            else:
                entry['venue'] = venue
        
        return entry
    
    def _extract_authors(self, text: str) -> Optional[str]:
        """Extract author names from text"""
        for pattern in self.author_patterns:
            match = pattern.search(text)
            if match:
                authors = match.group(1)
                # Clean up
                authors = authors.strip().rstrip('.')
                return authors
        return None
    
    def _extract_year(self, text: str) -> Optional[str]:
        """Extract publication year from text"""
        for pattern in self.year_patterns:
            match = pattern.search(text)
            if match:
                year = match.group(1)
                # Validate year is reasonable (1900-2100)
                if 1900 <= int(year) <= 2100:
                    return year
        return None
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract paper title from text"""
        for pattern in self.title_patterns:
            match = pattern.search(text)
            if match:
                title = match.group(1).strip()
                # Clean up common issues
                title = title.strip('.,;:')
                if len(title) > 10:  # Minimum reasonable title length
                    return title
        return None
    
    def _extract_venue(self, text: str) -> Optional[str]:
        """Extract venue (journal/conference) from text"""
        for pattern in self.venue_patterns:
            match = pattern.search(text)
            if match:
                venue = match.group(1).strip()
                venue = venue.strip('.,;:')
                return venue
        return None
    
    def get_all_citations(self, latex_content: str) -> List[str]:
        """
        Find all \\cite{...} commands in LaTeX
        
        Handles:
        - \\cite{key}
        - \\cite{key1,key2,key3}
        - \\citep{key}
        - \\citet{key}
        - \\cite[page]{key}
        
        Args:
            latex_content: LaTeX content
        
        Returns:
            List of unique citation keys
        """
        citation_keys = set()
        
        # Pattern to match various cite commands
        cite_patterns = [
            re.compile(r'\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]+)\}'),
            re.compile(r'\\citeauthor\*?(?:\[[^\]]*\])?\{([^}]+)\}'),
            re.compile(r'\\citeyear\*?(?:\[[^\]]*\])?\{([^}]+)\}'),
            re.compile(r'\\citealp\*?(?:\[[^\]]*\])?\{([^}]+)\}'),
        ]
        
        for pattern in cite_patterns:
            for match in pattern.finditer(latex_content):
                keys_str = match.group(1)
                # Handle multiple keys separated by comma
                keys = [k.strip() for k in keys_str.split(',')]
                citation_keys.update(keys)
        
        return list(citation_keys)
    
    def save_to_bib_file(self, output_path):
        """
        Save all BibTeX entries to .bib file
        
        Args:
            output_path: Path to output .bib file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        db = BibDatabase()
        db.entries = list(self.bibtex_entries.values())
        
        writer = BibTexWriter()
        writer.indent = '  '
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(writer.write(db))
    
    def merge_entries(self, entries1: Dict, entries2: Dict) -> Dict:
        """
        Merge two sets of BibTeX entries
        
        Args:
            entries1: First set of entries
            entries2: Second set of entries
        
        Returns:
            Merged entries (entries2 overwrites entries1 on conflicts)
        """
        merged = entries1.copy()
        merged.update(entries2)
        return merged
    
    def extract_all_from_directory(self, tex_dir) -> Dict:
        """
        Extract all references from a directory of LaTeX files
        
        Args:
            tex_dir: Directory containing .tex and .bib files
        
        Returns:
            Combined BibTeX entries
        """
        tex_dir = Path(tex_dir)
        all_entries = {}
        
        # First, check for .bib files
        for bib_file in tex_dir.glob('**/*.bib'):
            entries = self.extract_from_bib_file(bib_file)
            all_entries.update(entries)
        
        # Then, extract from .tex files (may override or add new)
        for tex_file in tex_dir.glob('**/*.tex'):
            try:
                with open(tex_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for thebibliography environment
                if r'\begin{thebibliography}' in content:
                    entries = self.extract_from_bibitem(content)
                    all_entries.update(entries)
            except Exception:
                continue
        
        self.bibtex_entries = all_entries
        return all_entries