"""
Unit tests for parser module

Tests cover:
- FileGatherer: multi-file gathering, \\input/\\include handling
- HierarchyBuilder: hierarchy construction, section/sentence/formula parsing
- LaTeXCleaner: cleanup and standardization
- ReferenceExtractor: \\bibitem parsing, BibTeX extraction
- Deduplicator: reference and content deduplication

Run with: pytest tests/test_parser.py -v
"""

import pytest
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from parser.file_gatherer import FileGatherer
from parser.hierarchy_builder import HierarchyBuilder
from parser.latex_cleaner import LaTeXCleaner
from parser.reference_extractor import ReferenceExtractor
from parser.deduplicator import Deduplicator


class TestFileGatherer:
    """Tests for FileGatherer class"""
    
    @pytest.fixture
    def temp_tex_dir(self):
        """Create temporary directory with test LaTeX files"""
        tmpdir = tempfile.mkdtemp()
        
        # Create main.tex
        main_tex = os.path.join(tmpdir, 'main.tex')
        with open(main_tex, 'w') as f:
            f.write(r'''
\documentclass{article}
\begin{document}
\input{sections/intro}
\include{sections/methods}
\end{document}
''')
        
        # Create sections directory
        sections_dir = os.path.join(tmpdir, 'sections')
        os.makedirs(sections_dir)
        
        # Create intro.tex
        with open(os.path.join(sections_dir, 'intro.tex'), 'w') as f:
            f.write(r'\section{Introduction}\nThis is the introduction.')
        
        # Create methods.tex
        with open(os.path.join(sections_dir, 'methods.tex'), 'w') as f:
            f.write(r'\section{Methods}\nThis is the methods section.')
        
        yield tmpdir
        
        # Cleanup
        shutil.rmtree(tmpdir)
    
    def test_find_main_file(self, temp_tex_dir):
        """Test main file detection"""
        gatherer = FileGatherer(temp_tex_dir)
        main_file = gatherer.find_main_file(temp_tex_dir)
        
        assert main_file is not None
        assert 'main.tex' in str(main_file)
    
    def test_find_tex_files(self, temp_tex_dir):
        """Test finding all .tex files"""
        gatherer = FileGatherer(temp_tex_dir)
        tex_files = gatherer.find_tex_files(temp_tex_dir)
        
        assert len(tex_files) >= 3  # main.tex, intro.tex, methods.tex
    
    def test_resolve_input_files(self, temp_tex_dir):
        """Test resolving \\input commands"""
        gatherer = FileGatherer(temp_tex_dir)
        main_file = os.path.join(temp_tex_dir, 'main.tex')
        
        with open(main_file, 'r') as f:
            content = f.read()
        
        resolved = gatherer.resolve_input_files(content, temp_tex_dir)
        
        assert 'Introduction' in resolved
        assert 'Methods' in resolved


class TestHierarchyBuilder:
    """Tests for HierarchyBuilder class"""
    
    def test_parse_sections(self):
        """Test section parsing"""
        builder = HierarchyBuilder('test-paper')
        
        latex_content = r'''
\section{Introduction}
This is the introduction.

\section{Methods}
This is the methods section.

\subsection{Data Collection}
We collected data.
'''
        builder.parse_latex(latex_content, version=1)
        
        elements = builder.get_elements()
        hierarchy = builder.get_hierarchy()
        
        # Check elements were created
        assert len(elements) > 0
        
        # Check for section elements
        section_types = [e.get('type') for e in elements.values()]
        assert 'section' in section_types or any('section' in str(t).lower() for t in section_types)
    
    def test_parse_sentences(self):
        """Test sentence extraction"""
        builder = HierarchyBuilder('test-paper')
        
        latex_content = r'''
\section{Introduction}
This is the first sentence. This is the second sentence.
And here is a third sentence.
'''
        builder.parse_latex(latex_content, version=1)
        
        # Check that sentences were extracted
        elements = builder.get_elements()
        
        # At least some elements should have content
        contents = [e.get('content', e.get('text', '')) for e in elements.values()]
        assert any('sentence' in c.lower() for c in contents)
    
    def test_parse_formulas(self):
        """Test formula extraction"""
        builder = HierarchyBuilder('test-paper')
        
        latex_content = r'''
\section{Methods}
The famous equation is $E = mc^2$ and also:
\begin{equation}
\frac{\partial f}{\partial x} = 0
\end{equation}
'''
        builder.parse_latex(latex_content, version=1)
        
        elements = builder.get_elements()
        
        # Check for formula elements
        types = [e.get('type', '') for e in elements.values()]
        has_formula = any('formula' in str(t).lower() or 'equation' in str(t).lower() or 'math' in str(t).lower() 
                         for t in types)
        
        # Or check content contains formulas
        contents = [str(e.get('content', '')) for e in elements.values()]
        has_math_content = any('mc^2' in c or 'partial' in c for c in contents)
        
        assert has_formula or has_math_content
    
    def test_parse_figures(self):
        """Test figure extraction"""
        builder = HierarchyBuilder('test-paper')
        
        latex_content = r'''
\section{Results}
\begin{figure}[h]
\includegraphics{fig1.png}
\caption{This is figure 1.}
\label{fig:1}
\end{figure}
'''
        builder.parse_latex(latex_content, version=1)
        
        elements = builder.get_elements()
        
        # Check for figure elements
        types = [str(e.get('type', '')) for e in elements.values()]
        contents = [str(e.get('content', '')) + str(e.get('caption', '')) for e in elements.values()]
        
        has_figure = any('figure' in t.lower() for t in types) or \
                    any('figure 1' in c.lower() or 'fig1' in c.lower() for c in contents)
        
        assert has_figure
    
    def test_generate_ids(self):
        """Test unique ID generation"""
        builder = HierarchyBuilder('test-2304.12345')
        
        latex_content = r'''
\section{A}
Content A.
\section{B}
Content B.
'''
        builder.parse_latex(latex_content, version=1)
        
        elements = builder.get_elements()
        
        # All IDs should be unique
        ids = list(elements.keys())
        assert len(ids) == len(set(ids))
        
        # IDs should contain the arxiv ID
        for elem_id in ids:
            assert 'test-2304.12345' in elem_id or '2304' in elem_id
    
    def test_get_hierarchy_json(self):
        """Test JSON export"""
        builder = HierarchyBuilder('test-paper')
        
        latex_content = r'\section{Test}\nContent.'
        builder.parse_latex(latex_content, version=1)
        
        hierarchy_json = builder.get_hierarchy_json()
        
        assert 'elements' in hierarchy_json
        assert isinstance(hierarchy_json['elements'], dict)


class TestLaTeXCleaner:
    """Tests for LaTeXCleaner class"""
    
    def test_remove_formatting(self):
        """Test formatting command removal"""
        cleaner = LaTeXCleaner()
        
        text = r"\textbf{bold} \textit{italic} \centering text"
        cleaned = cleaner.clean(text)
        
        assert r'\textbf' not in cleaned
        assert r'\textit' not in cleaned
        assert r'\centering' not in cleaned
        assert 'bold' in cleaned
        assert 'italic' in cleaned
    
    def test_remove_positioning(self):
        """Test positioning placeholder removal"""
        cleaner = LaTeXCleaner()
        
        text = r"Some text [htpb] and more [h!] stuff"
        cleaned = cleaner.clean(text)
        
        assert '[htpb]' not in cleaned
        assert '[h!]' not in cleaned
    
    def test_remove_table_commands(self):
        """Test table command removal"""
        cleaner = LaTeXCleaner()
        
        text = r"\hline \midrule \toprule \bottomrule"
        cleaned = cleaner.clean(text)
        
        assert r'\hline' not in cleaned
        assert r'\midrule' not in cleaned
        assert r'\toprule' not in cleaned
        assert r'\bottomrule' not in cleaned
    
    def test_normalize_math(self):
        """Test math normalization"""
        cleaner = LaTeXCleaner()
        
        # Inline math should be preserved
        text = r"The equation $E = mc^2$ is important."
        cleaned = cleaner.clean(text)
        
        # Math content should still be there (possibly in different form)
        assert 'E' in cleaned or 'mc' in cleaned
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization"""
        cleaner = LaTeXCleaner()
        
        text = "Multiple    spaces   and\n\n\nnewlines"
        cleaned = cleaner.clean(text)
        
        # Should not have multiple consecutive spaces or newlines
        assert '    ' not in cleaned
        assert '\n\n\n' not in cleaned


class TestReferenceExtractor:
    """Tests for ReferenceExtractor class"""
    
    def test_extract_bibitem_basic(self):
        """Test basic \\bibitem extraction"""
        extractor = ReferenceExtractor()
        
        latex_content = r'''
\begin{thebibliography}{99}
\bibitem{lipton2018}
Lipton, Z. C. (2018). The mythos of model interpretability. ACM Queue.

\bibitem{rudin2019}
Rudin, C. (2019). Stop explaining black box models. Nature Machine Intelligence.
\end{thebibliography}
'''
        entries = extractor.extract_bibitems(latex_content)
        
        assert 'lipton2018' in entries
        assert 'rudin2019' in entries
    
    def test_extract_bibitem_fields(self):
        """Test field extraction from \\bibitem"""
        extractor = ReferenceExtractor()
        
        latex_content = r'''
\bibitem{smith2020}
Smith, John and Doe, Jane (2020). A Great Paper Title. Journal of Testing, 10(2), 100-200.
'''
        entries = extractor.extract_bibitems(latex_content)
        
        assert 'smith2020' in entries
        entry = entries['smith2020']
        
        # Should have extracted some fields
        assert entry.get('raw') or entry.get('title') or 'Smith' in str(entry)
    
    def test_extract_citations(self):
        """Test \\cite command extraction"""
        extractor = ReferenceExtractor()
        
        text = r'''
As shown in \cite{lipton2018}, and also \cite{rudin2019,smith2020}.
Furthermore, \citep{jones2021} and \citet{wang2022} also demonstrated this.
'''
        citations = extractor.extract_citations(text)
        
        assert 'lipton2018' in citations
        assert 'rudin2019' in citations
        assert 'smith2020' in citations
    
    def test_parse_author_string(self):
        """Test author string parsing"""
        extractor = ReferenceExtractor()
        
        # Test various formats
        author_str = "Smith, John and Doe, Jane and Johnson, Bob"
        
        # The extractor should handle this format
        # Implementation may vary


class TestDeduplicator:
    """Tests for Deduplicator class"""
    
    def test_find_duplicates_exact(self):
        """Test exact duplicate detection"""
        deduper = Deduplicator(title_threshold=95)
        
        entries = {
            'lipton2018a': {'title': 'The mythos of model interpretability', 'year': '2018'},
            'lipton2018b': {'title': 'The mythos of model interpretability', 'year': '2018'},
        }
        
        duplicates = deduper._find_duplicates(entries)
        
        # Should find these as duplicates
        assert any(len(group) > 1 for group in duplicates)
    
    def test_find_duplicates_similar(self):
        """Test similar title detection"""
        deduper = Deduplicator(title_threshold=90)
        
        entries = {
            'lipton2018a': {'title': 'The Mythos of Model Interpretability', 'year': '2018'},
            'lipton2018b': {'title': 'The mythos of model interpretability', 'year': '2018'},
        }
        
        # Check if duplicate detection works
        result = deduper._is_duplicate(entries['lipton2018a'], entries['lipton2018b'])
        assert result == True
    
    def test_choose_canonical_key(self):
        """Test canonical key selection"""
        deduper = Deduplicator()
        
        entries = {
            'verylongkey_2018_paper_v1': {'title': 'Test', 'year': '2018'},
            'lipton2018': {'title': 'Test', 'year': '2018', 'author': 'Lipton'},
        }
        
        canonical = deduper._choose_canonical_key(
            ['verylongkey_2018_paper_v1', 'lipton2018'],
            entries
        )
        
        # Should prefer shorter, more standard key
        assert canonical == 'lipton2018'
    
    def test_unionize_fields(self):
        """Test field unionization"""
        deduper = Deduplicator()
        
        entries = {
            'key1': {'title': 'Paper Title', 'year': '2018', 'venue': ''},
            'key2': {'title': 'Paper Title', 'year': '', 'venue': 'ICML'},
        }
        
        merged = deduper._unionize_fields(['key1', 'key2'], entries)
        
        assert merged.get('title') == 'Paper Title'
        assert merged.get('year') == '2018'
        assert merged.get('venue') == 'ICML'
    
    def test_deduplicate_content(self):
        """Test content deduplication"""
        deduper = Deduplicator()
        
        elements = {
            'elem1': 'This is some content.',
            'elem2': 'This is some content.',  # Duplicate
            'elem3': 'Different content here.',
        }
        
        deduped, mapping = deduper.deduplicate_content(elements)
        
        # elem1 and elem2 should map to same ID
        assert mapping['elem1'] == mapping['elem2']
        
        # elem3 should be different
        assert mapping['elem3'] != mapping['elem1']


class TestIntegration:
    """Integration tests for parser pipeline"""
    
    @pytest.fixture
    def sample_latex(self):
        """Sample LaTeX document"""
        return r'''
\documentclass{article}
\title{A Sample Paper}
\author{John Doe}

\begin{document}
\maketitle

\section{Introduction}
This is the introduction. We cite \cite{lipton2018} here.

\section{Methods}
The equation is $E = mc^2$.

\begin{figure}[h]
\centering
\includegraphics{fig1.png}
\caption{A sample figure.}
\end{figure}

\begin{thebibliography}{99}
\bibitem{lipton2018}
Lipton, Z. C. (2018). The mythos of model interpretability.
\end{thebibliography}

\end{document}
'''
    
    def test_full_pipeline(self, sample_latex):
        """Test full parsing pipeline"""
        # Clean
        cleaner = LaTeXCleaner()
        cleaned = cleaner.clean(sample_latex)
        
        # Build hierarchy
        builder = HierarchyBuilder('test-paper')
        builder.parse_latex(cleaned, version=1)
        
        # Extract references
        extractor = ReferenceExtractor()
        refs = extractor.extract_bibitems(sample_latex)
        
        # Verify results
        elements = builder.get_elements()
        assert len(elements) > 0
        assert len(refs) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])