"""
Extract and convert references to BibTeX format
Handles \\bibitem and existing .bib files

As per requirement 2.1.3:
- For citations defined using \\bibitem within .tex files, convert them 
  into standard BibTeX entries using programmatic tools (e.g., Regular Expressions)
  
Extracts references from:
1. \bibitem entries in .tex files
2. Existing .bib files
3. Multiple versions
Converts all to standard BibTeX format
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# CRITICAL: Increase recursion limit before importing bibtexparser
# bibtexparser uses pyparsing which can cause RecursionError with complex files
if sys.getrecursionlimit() < 10000:
    sys.setrecursionlimit(10000)

# Disable pyparsing packrat to prevent recursion issues
try:
    import pyparsing
    pyparsing.ParserElement.disablePackrat()
except (ImportError, AttributeError):
    pass

import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase


class ReferenceExtractor:
    """
    Extract and convert references to BibTeX format
    
    IMPROVED VERSION:
    - Scans .bib, .tex, and .bbl files
    - Better regex patterns for \\bibitem variations
    - Handles encoding issues gracefully
    - Supports more citation formats
    """
    
    def __init__(self):
        self.bibtex_entries = {}
        self.extraction_stats = {
            'bibitem_count': 0,
            'bib_file_count': 0,
            'total_unique': 0
        }
        
        # Encodings to try when reading files
        self.encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
        
        # Patterns for parsing bibitem content (IMPROVED for better coverage)
        self.author_patterns = [
            # "Author, A. and Author, B." - standard format
            re.compile(r'^([A-Z][a-zàéèêëâäùûüîïôö\'-]+(?:,?\s+[A-Z]\.?)*(?:\s*,?\s*(?:and|&)\s+[A-Z][a-zàéèêëâäùûüîïôö\'-]+(?:,?\s+[A-Z]\.?)*)*)', re.IGNORECASE),
            # "A. Author, B. Author" - initials first
            re.compile(r'^([A-Z]\.?\s*[A-Z][a-z]+(?:\s*,?\s*(?:and|&)?\s*[A-Z]\.?\s*[A-Z][a-z]+)*)', re.IGNORECASE),
            # "Last, First and Last, First" - comma separated format
            re.compile(r'^([A-Z][a-z]+,\s*[A-Z]\.?(?:\s*(?:and|&)\s*[A-Z][a-z]+,\s*[A-Z]\.?)*)', re.IGNORECASE),
            # Generic: Capture everything until we see a clear title/venue indicator
            re.compile(r'^(.+?)(?=\.\s*["\u201c`]|\.?\s*(?:In|Proc|Journal|Trans|\d{4}))', re.IGNORECASE),
        ]
        
        self.year_patterns = [
            re.compile(r'\((\d{4})\)'),      # (2020)
            re.compile(r',\s*(\d{4})\.'),    # , 2020.
            re.compile(r'(\d{4})\.'),        # 2020.
            re.compile(r'\b(\d{4})\b'),      # any 4-digit year
        ]
        
        self.title_patterns = [
            # Title in quotes: "Title" or ``Title''
            re.compile(r'[""``]([^"``\'\']+)[""\'\']+'),
            # Title between author and venue
            re.compile(r'\.?\s+([^.]+?)\.\s+(?:In\s+|Proceedings|Journal|Conference|arXiv)', re.IGNORECASE),
        ]
        
        self.venue_patterns = [
            re.compile(r'In\s+(?:Proceedings\s+of\s+(?:the\s+)?)?([^.]+?)(?:,|\d{4}|\.)', re.IGNORECASE),
            re.compile(r'(?:Journal|Transactions)\s+(?:of|on)\s+([^.]+?)(?:,|\d{4}|\.)', re.IGNORECASE),
            re.compile(r'arXiv(?:\s+preprint)?[:\s]*([^\s,]+)', re.IGNORECASE),
        ]
    
    def _read_file_with_fallback_encoding(self, file_path: Path) -> str:
        """
        Read file content with fallback encodings
        
        Tries multiple encodings to handle files with different character sets.
        This fixes the issue where parser stops due to encoding errors.
        
        Args:
            file_path: Path to file
            
        Returns:
            File content as string
        """
        file_path = Path(file_path)
        
        for encoding in self.encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='strict') as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # For other errors, just continue to next encoding
                continue
        
        # Last resort: read with errors='ignore'
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return ""
    
    def extract_all_from_directory(self, tex_dir: Path) -> Dict:
        """
        Extract ALL references from a directory
        This is the MAIN entry point
        
        OPTIMIZED per Lab 2 requirement 2.1.1:
        - Only parse files actually included in final PDF
        - Skip unused/redundant files
        - For large .bib files, only extract entries that are actually cited
        
        IMPROVED for better coverage:
        - Also scan .bbl files (compiled bibliography)
        - Better regex for various \bibitem formats
        - Handle encoding issues gracefully
        
        Args:
            tex_dir: Directory containing .tex and .bib files
            
        Returns:
            Dict of BibTeX entries {key: entry_dict}
        """
        tex_dir = Path(tex_dir)
        all_entries = {}
        
        print(f"\n🔍 Extracting references from: {tex_dir}")
        
        # Step 0: First, collect all citation keys from \cite{} commands in .tex files
        # This allows us to filter large .bib files to only include used references
        tex_files = list(tex_dir.glob('**/*.tex'))
        bbl_files = list(tex_dir.glob('**/*.bbl'))  # Also scan .bbl files!
        print(f"   Found {len(tex_files)} .tex files, {len(bbl_files)} .bbl files")
        
        cited_keys = set()
        all_tex_content = ""
        for tex_file in tex_files:
            try:
                # Use fallback encoding for better compatibility
                content = self._read_file_with_fallback_encoding(tex_file)
                if content:
                    all_tex_content += content + "\n"
                    # Collect citation keys
                    keys = self.get_all_citations(content)
                    cited_keys.update(keys)
            except Exception as e:
                print(f"   ✗ Error reading {tex_file.name}: {e}")
        
        print(f"   Found {len(cited_keys)} unique citation keys in \\cite{{}} commands")
        
        # Step 1: Find and parse .bib files
        bib_files = list(tex_dir.glob('**/*.bib'))
        print(f"   Found {len(bib_files)} .bib files")
        
        # Size limit for full parsing (files larger than this use selective extraction)
        MAX_BIB_SIZE_FULL_PARSE = 2 * 1024 * 1024  # 2 MB
        
        for bib_file in bib_files:
            try:
                file_size = bib_file.stat().st_size
                
                if file_size > MAX_BIB_SIZE_FULL_PARSE:
                    # Large file: use selective extraction based on cited keys
                    print(f"   ⚡ {bib_file.name}: large file ({file_size / 1024 / 1024:.1f} MB), using selective extraction...")
                    entries = self._extract_cited_entries_from_large_bib(bib_file, cited_keys)
                    print(f"   ✓ {bib_file.name}: extracted {len(entries)} cited entries")
                else:
                    # Normal file: parse everything
                    entries = self.extract_from_bib_file(bib_file)
                    print(f"   ✓ {bib_file.name}: {len(entries)} entries")
                
                all_entries.update(entries)
                self.extraction_stats['bib_file_count'] += 1
            except Exception as e:
                print(f"   ✗ Error reading {bib_file.name}: {e}")
        
        # Step 2: Parse .tex files for \bibitem (reuse content already loaded)
        # IMPORTANT: Scan ALL .tex files for \bibitem, not just those with \begin{thebibliography}
        # Many papers have bibitems in separate files (refs.tex, biblio.tex) without the wrapper
        
        bibitem_found = False
        # Check if any \bibitem exists in the combined content
        if r'\bibitem' in all_tex_content:
            entries = self.extract_from_bibitem(all_tex_content)
            if entries:
                print(f"   ✓ Found {len(entries)} bibitems in .tex files")
                all_entries.update(entries)
                bibitem_found = True
                self.extraction_stats['bibitem_count'] += len(entries)
        
        # Step 3: Also check .bbl files (compiled bibliography)
        # These contain \bibitem entries in compiled form
        for bbl_file in bbl_files:
            try:
                bbl_content = self._read_file_with_fallback_encoding(bbl_file)
                if r'\bibitem' in bbl_content:
                    entries = self.extract_from_bibitem(bbl_content)
                    if entries:
                        print(f"   ✓ {bbl_file.name}: {len(entries)} bibitems")
                        all_entries.update(entries)
                        bibitem_found = True
                        self.extraction_stats['bibitem_count'] += len(entries)
            except Exception as e:
                print(f"   ✗ Error reading {bbl_file.name}: {e}")
        
        if not bibitem_found and len(bib_files) == 0:
            print("   ⚠️  WARNING: No .bib files or \\bibitem entries found!")
        
        self.bibtex_entries = all_entries
        self.extraction_stats['total_unique'] = len(all_entries)
        
        print(f"\n📊 Extraction summary:")
        print(f"   Total unique entries: {len(all_entries)}")
        print(f"   From .bib files: {self.extraction_stats['bib_file_count']} files")
        print(f"   From \\bibitem: {self.extraction_stats['bibitem_count']} entries")
        
        return all_entries
    
    def extract_from_bib_file(self, bib_file_path: Path) -> Dict:
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
        
        try:
            with open(bib_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                bib_database = bibtexparser.load(f)
            
            entries = {}
            for entry in bib_database.entries:
                key = entry.get('ID', '')
                if key:
                    entries[key] = entry
            
            return entries
        except Exception as e:
            print(f"Error parsing {bib_file_path}: {e}")
            return {}
    
    def _extract_cited_entries_from_large_bib(self, bib_file_path: Path, cited_keys: set) -> Dict:
        """
        Extract only cited entries from a large .bib file using regex
        This avoids loading the entire file into bibtexparser which can be slow
        
        Per Lab 2 requirement 2.1.1: Only parse files actually used in final PDF
        
        Args:
            bib_file_path: Path to large .bib file
            cited_keys: Set of citation keys that are actually used in .tex files
            
        Returns:
            Dict of BibTeX entries {key: entry_dict} for cited entries only
        """
        if not cited_keys:
            return {}
        
        entries = {}
        
        try:
            with open(bib_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Pattern to match BibTeX entries: @type{key, ... }
            # This regex finds entry boundaries
            entry_pattern = re.compile(
                r'@(\w+)\s*\{\s*([^,\s]+)\s*,([^@]*?)(?=\n@|\Z)',
                re.DOTALL | re.MULTILINE
            )
            
            for match in entry_pattern.finditer(content):
                entry_type = match.group(1).lower()
                entry_key = match.group(2).strip()
                entry_body = match.group(3)
                
                # Only process if this key is actually cited
                if entry_key in cited_keys:
                    # Parse the entry body to extract fields
                    entry = {
                        'ID': entry_key,
                        'ENTRYTYPE': entry_type,
                    }
                    
                    # Extract fields using regex
                    field_pattern = re.compile(r'(\w+)\s*=\s*[{"]([^}"]*)[}"]', re.DOTALL)
                    for field_match in field_pattern.finditer(entry_body):
                        field_name = field_match.group(1).lower()
                        field_value = field_match.group(2).strip()
                        entry[field_name] = field_value
                    
                    entries[entry_key] = entry
            
            return entries
            
        except Exception as e:
            print(f"Error extracting from large bib {bib_file_path}: {e}")
            return {}
    
    def extract_from_bibitem(self, latex_content: str) -> Dict:
        """
        Extract \\bibitem entries and convert to BibTeX
        
        Pattern:
            \\bibitem{key}
            Author et al. Title. Journal, year.
        
        IMPROVED COVERAGE:
        - \\bibitem{key} - standard
        - \\bibitem {key} - with space
        - \\bibitem[label]{key} - with optional label
        - \\bibitem[label] {key} - label + space
        
        Args:
            latex_content: LaTeX content containing bibliography
            
        Returns:
            Dict of BibTeX entries {key: entry_dict}
        """
        entries = {}
        
        # Multiple patterns to catch all variations of \bibitem
        # Pattern 1: Standard \bibitem{key} or \bibitem{key}
        # Pattern 2: \bibitem[label]{key} with optional label
        # Pattern 3: Handle space variations: \bibitem {key}, \bibitem[ label ] { key }
        bibitem_patterns = [
            # Most specific: with optional label [], handles spaces
            re.compile(
                r'\\bibitem\s*(?:\[[^\]]*\])?\s*\{\s*([^}]+?)\s*\}\s*\n?(.*?)(?=\\bibitem|\\end\{thebibliography\}|\\end\{document\}|\\begin\{thebibliography\}|\\section|\\chapter|$)',
                re.DOTALL
            ),
            # Alternative pattern for edge cases: key may contain special chars
            re.compile(
                r'\\bibitem\s*(?:\[[^\]]*\])?\s*\{([A-Za-z0-9_:.-]+)\}\s*(.*?)(?=\\bibitem|\\end|$)',
                re.DOTALL
            ),
        ]
        
        found_keys = set()  # Track found keys to avoid duplicates
        
        for pattern in bibitem_patterns:
            for match in pattern.finditer(latex_content):
                key = match.group(1).strip()
                bibitem_text = match.group(2).strip()
                
                # Skip if already found or empty
                if key in found_keys:
                    continue
                if not bibitem_text or len(bibitem_text) < 5:
                    continue
                
                # Clean up key (remove any remaining whitespace/special chars at ends)
                key = re.sub(r'^[\s{}]+|[\s{}]+$', '', key)
                
                if not key:
                    continue
                
                found_keys.add(key)
                
                # Convert to BibTeX entry
                entry = self.bibitem_to_bibtex(key, bibitem_text)
                entries[key] = entry
        
        return entries
    
    def bibitem_to_bibtex(self, key: str, bibitem_text: str) -> Dict:
        """
        Convert a single \\bibitem to BibTeX format
        
        COMPLETELY REWRITTEN to handle common formats:
        
        Format 1: "Author1, Author2, Title, \\href{DOI}{Journal info (Year)}."
        Format 2: "Author1 and Author2. Title. Journal, Year."
        Format 3: "Author1, Title, Journal Volume, Pages (Year)."
        
        Args:
            key: Citation key
            bibitem_text: Text content after \\bibitem{key}
            
        Returns:
            BibTeX entry dict
        """
        # Clean up the text first
        text = bibitem_text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove LaTeX comment lines (important: do this early!)
        text = re.sub(r'%[^\n]*', '', text)
        text = re.sub(r'%.*$', '', text)  # Also handle end-of-line comments
        
        # Clean tildes (non-breaking spaces in LaTeX)
        text = text.replace('~', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        entry = {
            'ID': key,
            'ENTRYTYPE': 'misc',  # Default type
        }
        
        # ===== PRE-PROCESSING: Extract DOI before removing \href =====
        # This preserves the DOI information while cleaning the text
        doi_extracted = None
        href_match = re.search(
            r'\\href\{(https?://doi\.org/[^}]+)\}\s*\{([^}]+)\}',
            text
        )
        
        if href_match:
            doi_url = href_match.group(1)
            journal_info = href_match.group(2).strip()
            
            # Extract DOI
            doi_match = re.search(r'doi\.org/(.+)$', doi_url)
            if doi_match:
                doi_extracted = doi_match.group(1)
                entry['doi'] = doi_extracted
            
            # Parse journal info: "Journal Name Volume, Pages (Year)"
            journal_parsed = self._parse_journal_info(journal_info)
            if journal_parsed:
                entry.update(journal_parsed)
            
            # Get the text BEFORE \href - this contains Author, Title
            before_href = text[:text.find('\\href')].strip()
            before_href = before_href.rstrip(',').strip()
            
            # Parse "Author1, Author2, ..., Title" format
            parsed = self._parse_author_title_before_href(before_href)
            if parsed:
                if 'author' in parsed:
                    entry['author'] = parsed['author']
                if 'title' in parsed:
                    entry['title'] = parsed['title']
            
            # Determine entry type based on journal
            if 'journal' in entry:
                journal_lower = entry['journal'].lower()
                if any(x in journal_lower for x in ['proc', 'conf', 'workshop', 'sympos']):
                    entry['ENTRYTYPE'] = 'inproceedings'
                    entry['booktitle'] = entry.pop('journal')
                else:
                    entry['ENTRYTYPE'] = 'article'
        
        # ===== STRATEGY 2: Standard format without \href =====
        else:
            # First, clean up any raw DOI links that might interfere
            text_clean = re.sub(r'https?://doi\.org/[^\s}]+', '', text)
            text_clean = re.sub(r'\\doi\{[^}]*\}', '', text_clean)
            text_clean = re.sub(r'\s+', ' ', text_clean).strip()
            
            # Try to parse: "Author. Title. Venue, Year." or "Author, Title, Venue (Year)."
            parsed = self._parse_standard_bibitem(text_clean if text_clean else text)
            entry.update(parsed)
        
        # ===== FINAL CLEANUP =====
        # Clean all string fields
        for field in ['author', 'title', 'journal', 'booktitle', 'venue']:
            if field in entry and entry[field]:
                entry[field] = self._clean_bibtex_value(entry[field])
        
        # Validate title - don't use raw text fallback if it would be garbage
        # Instead, return None for title if we can't parse it properly
        if 'title' not in entry or not entry.get('title'):
            # Only use fallback if text looks like it contains a valid title
            # (not just author names or journal info)
            fallback_title = self._extract_title_fallback(text)
            if fallback_title:
                entry['title'] = fallback_title
            # If still no title, leave it empty rather than filling with garbage
            # This is better for matching quality - missing data is better than wrong data
        
        return entry
    
    def _extract_title_fallback(self, text: str) -> Optional[str]:
        """
        Extract title using fallback strategies - more conservative than before.
        Returns None if no valid title can be extracted.
        
        This avoids the problem of filling title with garbage data.
        """
        # Clean the text first
        text = self._clean_bibtex_value(text)
        
        if not text or len(text) < 10:
            return None
        
        # Strategy 1: Look for text in quotes
        quoted_match = re.search(r'[""``]([^""``\'\']{10,200})[""\'\']+', text)
        if quoted_match:
            return quoted_match.group(1).strip()
        
        # Strategy 2: Look for text between periods that's title-like
        # (not too short, not starting with initials)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        for i, sentence in enumerate(sentences):
            # Skip first sentence (usually author)
            if i == 0:
                continue
            # Skip sentences that look like author names (has initials)
            if re.match(r'^[A-Z]\.\s*[A-Z]', sentence):
                continue
            # Skip sentences that look like journal info (has volume/page numbers)
            if re.search(r'\d+\s*,\s*\d+', sentence):
                continue
            # Valid title candidate
            if 10 < len(sentence) < 200:
                return sentence.strip('.,;: ')
        
        # Strategy 3: If text is reasonable length and doesn't look like metadata, use it
        # But be very conservative - reject if it looks like garbage
        if 20 < len(text) < 200:
            # Reject if mostly numbers or special characters
            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            if alpha_ratio < 0.5:
                return None
            # Reject if it looks like it contains DOI/URL fragments
            if 'doi.org' in text.lower() or 'http' in text.lower():
                return None
            # Accept as title
            return text
        
        return None
    
    def _parse_journal_info(self, journal_info: str) -> Dict:
        """
        Parse journal information string
        
        Examples:
            "Am. J. Phys. 71, 1095 (2003)"
            "Phys. Rev. Lett. 128, 110402 (2022)"
            "Nat. Mater. 18, 783 (2019)"
            "Science 363, 6422 (2019)"
        
        Returns:
            Dict with journal, volume, pages, year
        """
        result = {}
        
        # Extract year from (YYYY)
        year_match = re.search(r'\((\d{4})\)', journal_info)
        if year_match:
            result['year'] = year_match.group(1)
        
        # Remove year part for further parsing
        text = re.sub(r'\s*\(\d{4}\)\s*\.?\s*$', '', journal_info).strip()
        
        # Pattern: "Journal Name Volume, Pages" or "Journal Name Volume, Article_ID"
        # Examples: "Am. J. Phys. 71, 1095", "Phys. Rev. Lett. 128, 110402"
        match = re.match(
            r'^(.+?)\s+(\d+)\s*,\s*(\d+(?:[-–]\d+)?|[A-Za-z]?\d+)$',
            text
        )
        
        if match:
            result['journal'] = match.group(1).strip()
            result['volume'] = match.group(2)
            result['pages'] = match.group(3)
        else:
            # Fallback: just use as journal name
            result['journal'] = text
        
        return result
    
    def _parse_author_title_before_href(self, text: str) -> Dict:
        """
        Parse "Author1, Author2, ..., Title" format
        
        The challenge: both authors and title are comma-separated
        Strategy: Title is usually the last comma-separated segment that:
        - Contains more than 3 words
        - Doesn't look like an author name (no initials pattern)
        - May contain question marks, colons, or other title-like punctuation
        
        Examples:
            "C.~M.~Bender, Must a Hamiltonian be Hermitian?"
            "M.~Naghiloo, M.~Abbasi, Y.~N.~Joglekar and K.~W.~Murch, Quantum state tomography"
        """
        result = {}
        
        if not text:
            return result
        
        # Clean tildes and normalize
        text = text.replace('~', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split by comma, but be careful with "and" in author lists
        # First, try to find where title starts
        
        # Strategy: Look for the LAST segment that looks like a title
        # Title indicators: question mark, colon after first word, or longer phrase
        
        # Pattern for author: starts with initial or short name, e.g., "C. M. Bender", "A. Smith"
        author_pattern = re.compile(
            r'^[A-Z]\.?\s*(?:[A-Z]\.?\s*)*[A-Z][a-z]+$|'  # "C. M. Bender" or "A. Smith"
            r'^[A-Z][a-z]+,?\s*[A-Z]\.?$|'                 # "Bender, C." or "Smith A."
            r'^[A-Z][a-z]+-[A-Z][a-z]+$'                   # "Perrey-Debain"
        )
        
        # Split by comma (but not comma within parentheses or braces)
        parts = self._smart_split(text, ',')
        
        if not parts:
            return result
        
        # Find where title starts - work backwards
        title_start_idx = len(parts)
        
        for i in range(len(parts) - 1, -1, -1):
            part = parts[i].strip()
            
            # Check if this looks like a title (not an author)
            is_title_like = (
                '?' in part or  # Question in title
                ':' in part or  # Colon often in title
                len(part.split()) > 4 or  # Long phrase
                (len(part.split()) > 2 and not self._looks_like_author(part))
            )
            
            if is_title_like:
                title_start_idx = i
                break
            
            # Check if it looks like an author name
            if self._looks_like_author(part):
                continue
            else:
                # Doesn't look like author, might be start of title
                title_start_idx = i
                break
        
        # Combine parts
        if title_start_idx < len(parts):
            # Authors are before title_start_idx
            author_parts = parts[:title_start_idx]
            title_parts = parts[title_start_idx:]
            
            if author_parts:
                authors = ', '.join(p.strip() for p in author_parts)
                # Clean up author string
                authors = self._clean_author_string(authors)
                if authors:
                    result['author'] = authors
            
            if title_parts:
                title = ', '.join(p.strip() for p in title_parts)
                title = title.strip('.,;: ')
                if title:
                    result['title'] = title
        else:
            # Couldn't split - use heuristic: first part is author, rest is title
            if len(parts) >= 2:
                result['author'] = self._clean_author_string(parts[0].strip())
                result['title'] = ', '.join(p.strip() for p in parts[1:]).strip('.,;: ')
            elif len(parts) == 1:
                # Single part - assume it's the title
                result['title'] = parts[0].strip()
        
        return result
    
    def _smart_split(self, text: str, delimiter: str) -> List[str]:
        """Split text by delimiter, but ignore delimiters inside braces/brackets"""
        parts = []
        current = ""
        depth = 0
        
        for char in text:
            if char in '{[(':
                depth += 1
                current += char
            elif char in '}])':
                depth -= 1
                current += char
            elif char == delimiter and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += char
        
        if current:
            parts.append(current)
        
        return parts
    
    def _looks_like_author(self, text: str) -> bool:
        """Check if text looks like an author name"""
        text = text.strip()
        
        if not text:
            return False
        
        # Common patterns for author names:
        # "C. M. Bender", "A. Smith", "Smith, A.", "Smith", "van der Berg"
        
        # Has initials pattern (letter followed by period)
        if re.search(r'\b[A-Z]\.\s*', text):
            return True
        
        # Short name (1-3 words, each capitalized)
        words = text.split()
        if 1 <= len(words) <= 3:
            # Check if all words start with capital or are connectors
            connectors = {'and', 'von', 'van', 'de', 'der', 'la', 'el'}
            valid = all(
                w[0].isupper() or w.lower() in connectors
                for w in words if w
            )
            if valid:
                return True
        
        # Ends with "et al"
        if re.search(r'\bet\s+al\.?\s*$', text, re.IGNORECASE):
            return True
        
        return False
    
    def _clean_author_string(self, authors: str) -> str:
        """Clean and normalize author string"""
        if not authors:
            return ""
        
        # Remove tildes
        authors = authors.replace('~', ' ')
        
        # Normalize spaces
        authors = re.sub(r'\s+', ' ', authors).strip()
        
        # Normalize "and" connectors
        authors = re.sub(r'\s*,\s*and\s+', ' and ', authors)
        authors = re.sub(r'\s+and\s+', ' and ', authors)
        authors = re.sub(r'\s*&\s*', ' and ', authors)
        
        # Remove trailing punctuation
        authors = authors.rstrip('.,;:')
        
        return authors
    
    def _parse_standard_bibitem(self, text: str) -> Dict:
        """
        Parse standard bibitem format (without href)
        
        Formats:
            "Author. Title. Venue, Year."
            "Author, Title, Venue (Year)."
            "Author, Title (Publisher, Year)."
        """
        result = {}
        
        # Clean text
        text = text.replace('~', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Extract year
        year = self._extract_year(text)
        if year:
            result['year'] = year
        
        # Special case: Book format "Author, Title (Publisher, City, Year)."
        book_match = re.match(
            r'^([A-Z]\.?\s*(?:[A-Z]\.?\s*)?[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z]\.?\s*(?:[A-Z]\.?\s*)?[A-Z][a-z]+)*),\s*'  # Authors
            r'(.+?)\s*'  # Title
            r'\(([^)]+,\s*\d{4})\)\s*\.?\s*$',  # (Publisher, City, Year)
            text,
            re.IGNORECASE
        )
        
        if book_match:
            result['author'] = self._clean_author_string(book_match.group(1))
            result['title'] = book_match.group(2).strip('.,;: ')
            pub_info = book_match.group(3)
            result['ENTRYTYPE'] = 'book'
            # Try to extract publisher
            pub_parts = pub_info.rsplit(',', 1)
            if len(pub_parts) >= 1:
                result['publisher'] = pub_parts[0].strip()
            return result
        
        # Try splitting by periods first (Author. Title. Venue.)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) >= 3:
            # Likely: Author. Title. Venue.
            result['author'] = self._clean_author_string(sentences[0])
            result['title'] = sentences[1].strip('.,;: ')
            
            # Rest might be venue
            venue = '. '.join(sentences[2:]).strip('.,;: ')
            if venue:
                result['journal'] = venue
        
        elif len(sentences) == 2:
            # Could be: "Author. Title and Venue." or "Author, Title. Venue."
            result['author'] = self._clean_author_string(sentences[0])
            result['title'] = sentences[1].strip('.,;: ')
        
        elif len(sentences) == 1:
            # Single sentence - try comma splitting for "Author, Title" format
            # Look for first comma that separates author from title
            text_clean = sentences[0]
            
            # Pattern: "G. Strang, Linear Algebra..." - author is short, title is long
            comma_idx = text_clean.find(',')
            if comma_idx > 0:
                potential_author = text_clean[:comma_idx].strip()
                potential_title = text_clean[comma_idx+1:].strip()
                
                # Check if potential_author looks like an author name
                if self._looks_like_author(potential_author):
                    result['author'] = self._clean_author_string(potential_author)
                    result['title'] = potential_title.strip('.,;: ')
                else:
                    # Fallback to full parsing
                    parsed = self._parse_author_title_before_href(text_clean)
                    result.update(parsed)
            else:
                result['title'] = text_clean.strip('.,;: ')
        
        return result
    
    def _clean_bibtex_value(self, value: str) -> str:
        """Clean a BibTeX field value"""
        if not value:
            return ""
        
        # Remove tildes
        value = value.replace('~', ' ')
        
        # Remove \href{...}{...} - keep the display text
        value = re.sub(r'\\href\{[^}]*\}\s*\{([^}]*)\}', r'\1', value)
        
        # Remove other LaTeX commands
        value = re.sub(r'\\emph\{([^}]*)\}', r'\1', value)
        value = re.sub(r'\\textit\{([^}]*)\}', r'\1', value)
        value = re.sub(r'\\textbf\{([^}]*)\}', r'\1', value)
        value = re.sub(r'\\textrm\{([^}]*)\}', r'\1', value)
        value = re.sub(r'\\text\{([^}]*)\}', r'\1', value)
        
        # Keep math mode content but mark it
        # value = re.sub(r'\$([^$]+)\$', r'$\1$', value)  # Keep as is
        
        # Remove comment markers
        value = re.sub(r'%[^\n]*', '', value)
        
        # Normalize whitespace
        value = re.sub(r'\s+', ' ', value).strip()
        
        # Remove leading/trailing punctuation (but keep internal)
        value = value.strip('.,;: ')
        
        return value
    
    def _extract_authors(self, text: str) -> Optional[str]:
        """Extract author names from text with improved fallback"""
        for pattern in self.author_patterns:
            match = pattern.search(text)
            if match:
                authors = match.group(1)
                # Clean up
                authors = authors.strip().rstrip('.')
                # Normalize "and" and "&"
                authors = re.sub(r',\s*(?:and|&)\s+', ' and ', authors)
                authors = re.sub(r'\s+(?:and|&)\s+', ' and ', authors)
                # Validate it looks like author names (has capital letters)
                if re.search(r'[A-Z]', authors) and len(authors) > 2:
                    return authors
        
        # Fallback: Extract text before first period as potential author
        first_period = text.find('.')
        if first_period > 5:
            potential_authors = text[:first_period].strip()
            # Clean LaTeX commands
            potential_authors = re.sub(r'\\[a-zA-Z]+(?:\{[^}]*\})?', '', potential_authors)
            potential_authors = potential_authors.strip()
            if potential_authors and len(potential_authors) > 2:
                return potential_authors
        
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
        
        # Fallback strategy 1: Try to extract text between first and second period
        # (typically: Author. Title. Venue)
        sentences = text.split('.')
        if len(sentences) >= 2:
            potential_title = sentences[1].strip()
            # Clean LaTeX commands
            potential_title = re.sub(r'\\[a-zA-Z]+(?:\{[^}]*\})?', '', potential_title)
            potential_title = potential_title.strip('.,;: ')
            if 10 < len(potential_title) < 200:
                return potential_title
        
        # Fallback strategy 2: Take everything after author pattern until venue/year
        # Find position after potential author names (letters followed by numbers could be year)
        author_end = re.search(r'(?:et\s+al\.?|and\s+[A-Z]\.|[A-Z]\.)\s*', text)
        if author_end:
            remaining = text[author_end.end():].strip()
            # Take text until we hit a venue indicator or year
            title_match = re.match(r'^([^.]+?)(?:\.|,\s*(?:In|Proc|Journal|Trans|\d{4}))', remaining, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip('.,;: ')
                if len(title) > 10:
                    return title
        
        # Fallback strategy 3: Use the entire content as raw reference
        # This ensures we don't lose data even if parsing fails
        if len(text) > 20:
            # Clean and truncate
            cleaned = re.sub(r'\s+', ' ', text).strip()
            if len(cleaned) > 300:
                cleaned = cleaned[:300] + '...'
            return cleaned
        
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
        - \\citep{key}, \\citet{key}
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
    
    def save_to_bib_file(self, output_path: Path):
        """
        Save all BibTeX entries to .bib file
        
        Args:
            output_path: Path to output .bib file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.bibtex_entries:
            print(f"⚠️  No entries to save to {output_path}")
            # Create empty file
            output_path.write_text("", encoding='utf-8')
            return
        
        db = BibDatabase()
        db.entries = list(self.bibtex_entries.values())
        
        writer = BibTexWriter()
        writer.indent = '  '
        writer.order_entries_by = None  # Keep original order
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(writer.write(db))
            
            print(f"✅ Saved {len(self.bibtex_entries)} entries to {output_path}")
        except Exception as e:
            print(f"❌ Error saving to {output_path}: {e}")
    
    def get_statistics(self) -> Dict:
        """Get extraction statistics"""
        return {
            'total_entries': len(self.bibtex_entries),
            'bibitem_count': self.extraction_stats['bibitem_count'],
            'bib_file_count': self.extraction_stats['bib_file_count'],
            'unique_entries': self.extraction_stats['total_unique']
        }