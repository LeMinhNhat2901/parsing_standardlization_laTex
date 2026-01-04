"""
File I/O utilities for JSON and BibTeX
Handles reading and writing data files with proper error handling
"""

import json
import os
from pathlib import Path
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase


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
    path = Path(path)
    ensure_dir(path.parent)
    
    db = BibDatabase()
    
    if isinstance(entries, dict):
        entries_list = []
        for key, entry in entries.items():
            entry_copy = entry.copy()
            entry_copy['ID'] = key
            entries_list.append(entry_copy)
        db.entries = entries_list
    else:
        db.entries = entries
    
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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BibTeX file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        bib_database = bibtexparser.load(f)
    
    # Convert to dictionary with ID as key
    entries = {}
    for entry in bib_database.entries:
        key = entry.get('ID', '')
        if key:
            entries[key] = entry
    
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