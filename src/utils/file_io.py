"""
File I/O utilities for JSON and BibTeX
Handles reading and writing data files with proper error handling
"""

import json
import os
from pathlib import Path

# Make bibtexparser optional to avoid import chain issues
try:
    import bibtexparser
    from bibtexparser.bwriter import BibTexWriter
    from bibtexparser.bibdatabase import BibDatabase
    HAS_BIBTEXPARSER = True
except ImportError:
    HAS_BIBTEXPARSER = False
    bibtexparser = None
    BibTexWriter = None
    BibDatabase = None


def load_json(path):
    """
    Load JSON file
    
    Args:
        path: Path to JSON file
    
    Returns:
        Parsed JSON data (dict or list)
    
    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data, indent=2):
    """
    Save data to JSON file with proper formatting
    
    Args:
        path: Path to save JSON file
        data: Data to save (dict or list)
        indent: Indentation level (default: 2)
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_bibtex(path, entries):
    """
    Save BibTeX entries to file
    
    Args:
        path: Path to save .bib file
        entries: Dict of entries {key: entry_dict} or list of entry dicts
    """
    if not HAS_BIBTEXPARSER:
        # Fallback: write simple bibtex format manually
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, 'w', encoding='utf-8') as f:
            # Use type() instead of isinstance() to avoid recursion issues
            if type(entries).__name__ == 'dict':
                for key, entry in entries.items():
                    entry_type = entry.get('ENTRYTYPE', 'misc')
                    f.write(f"@{entry_type}{{{key},\n")
                    for k, v in entry.items():
                        if k not in ['ID', 'ENTRYTYPE']:
                            f.write(f"  {k} = {{{v}}},\n")
                    f.write("}\n\n")
        return
    
    path = Path(path)
    ensure_dir(path.parent)
    
    db = BibDatabase()
    
    # Use type() instead of isinstance() to avoid recursion issues
    if type(entries).__name__ == 'dict':
        entries_list = []
        for key, entry in entries.items():
            entry_copy = dict(entry) if type(entry).__name__ == 'dict' else {}
            entry_copy['ID'] = key
            # CRITICAL: Ensure ENTRYTYPE is always present (required by bibtexparser)
            if 'ENTRYTYPE' not in entry_copy:
                entry_copy['ENTRYTYPE'] = 'misc'
            entries_list.append(entry_copy)
        db.entries = entries_list
    else:
        # Ensure ENTRYTYPE for list entries too
        db.entries = []
        for entry in entries:
            entry_copy = dict(entry) if type(entry).__name__ == 'dict' else {}
            if type(entry_copy).__name__ == 'dict' and 'ENTRYTYPE' not in entry_copy:
                entry_copy['ENTRYTYPE'] = 'misc'
            db.entries.append(entry_copy)
    
    writer = BibTexWriter()
    writer.indent = '  '
    writer.entry_separator = '\n'
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(writer.write(db))


def load_bibtex(path):
    """
    Load BibTeX file and parse entries
    
    Args:
        path: Path to .bib file
    
    Returns:
        Dict of BibTeX entries {key: entry_dict}
    """
    if not HAS_BIBTEXPARSER:
        # Return empty dict if bibtexparser not available
        return {}
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BibTeX file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        bib_database = bibtexparser.load(f)
    
    # Convert to dictionary with ID as key
    # IMPORTANT: Create clean copies to avoid bibtexparser internal objects
    # that can cause RecursionError with isinstance checks
    entries = {}
    for entry in bib_database.entries:
        key = entry.get('ID', '')
        if key:
            # Create a clean copy with only string values
            # Use type() instead of isinstance() to avoid recursion issues
            clean_entry = {}
            for k, v in entry.items():
                if v is None:
                    clean_entry[k] = ''
                elif type(v).__name__ == 'str':
                    clean_entry[k] = v
                elif type(v).__name__ in ('int', 'float', 'bool'):
                    clean_entry[k] = str(v)
                elif type(v).__name__ in ('list', 'tuple'):
                    clean_entry[k] = ', '.join(str(x) for x in v)
                else:
                    # Convert any other type to string
                    clean_entry[k] = str(v)
            entries[key] = clean_entry
    
    return entries


def ensure_dir(path):
    """
    Create directory if it does not exist
    
    Args:
        path: Directory path to create
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def read_tex_file(path):
    """
    Read LaTeX file with fallback encodings
    
    Args:
        path: Path to .tex file
    
    Returns:
        File content as string
    """
    path = Path(path)
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    raise UnicodeDecodeError(f"Could not decode file: {path}")


def write_tex_file(path, content):
    """
    Write LaTeX file
    
    Args:
        path: Path to save .tex file
        content: Content to write
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def list_files(directory, extension=None, recursive=True):
    """
    List files in directory
    
    Args:
        directory: Directory path
        extension: Filter by extension (e.g., '.tex')
        recursive: Whether to search recursively
    
    Returns:
        List of file paths
    """
    directory = Path(directory)
    
    if recursive:
        pattern = '**/*' + (extension or '')
    else:
        pattern = '*' + (extension or '')
    
    files = list(directory.glob(pattern))
    return [f for f in files if f.is_file()]


def combine_files(file_paths):
    """
    Combine multiple text files into one string
    
    Args:
        file_paths: List of file paths
    
    Returns:
        Combined content string
    """
    contents = []
    for path in file_paths:
        content = read_tex_file(path)
        contents.append(content)
    
    return '\n\n'.join(contents)