"""
Unit tests for parser module
"""

import pytest
from src.parser import *

def test_file_gatherer():
    """Test multi-file gathering"""
    gatherer = FileGatherer('./test_data/tex')
    main_file = gatherer.find_main_file('./test_data/tex/v1')
    assert main_file.exists()
    assert 'main.tex' in str(main_file)

def test_hierarchy_builder():
    """Test hierarchy construction"""
    builder = HierarchyBuilder('test-paper')
    latex_content = r"""
    \section{Introduction}
    This is a sentence.
    $$E = mc^2$$
    """
    builder.parse_latex(latex_content, version=1)
    
    assert len(builder.elements) > 0
    assert 'test-paper-sec-1' in builder.elements

def test_latex_cleaner():
    """Test LaTeX cleanup"""
    cleaner = LaTeXCleaner()
    
    text = r"\centering Some text [htpb] \midrule"
    cleaned = cleaner.clean(text)
    
    assert r'\centering' not in cleaned
    assert '[htpb]' not in cleaned
    assert r'\midrule' not in cleaned

def test_reference_extractor():
    """Test BibTeX extraction"""
    extractor = ReferenceExtractor()
    
    latex_content = r"""
    \bibitem{lipton2018}
    Lipton, Z. C. (2018). The mythos of model interpretability.
    """
    
    entries = extractor.extract_from_bibitem(latex_content)
    assert 'lipton2018' in entries

def test_deduplicator():
    """Test reference deduplication"""
    deduper = Deduplicator()
    
    entries = {
        'lipton2018a': {'title': 'The mythos of model interpretability'},
        'lipton2018b': {'title': 'The Mythos of Model Interpretability'},
    }
    
    duplicates = deduper._find_duplicates(entries)
    assert len(duplicates) > 0

if __name__ == '__main__':
    pytest.main([__file__])