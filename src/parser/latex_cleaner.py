"""
Clean and normalize LaTeX content
- Remove formatting commands that don't contribute to semantic meaning
- Normalize math notation (inline to $...$, block to equation environment)
- Standardize whitespace

As per requirement 2.1.3:
- Remove unnecessary formatting commands (e.g., \\centering, [htpb], \\midrule)
- Convert all inline math to $...$
- Convert all block math to \\begin{equation}...\\end{equation}
"""

import re
import sys
from typing import List

# CRITICAL: Increase recursion limit for complex regex processing
if sys.getrecursionlimit() < 10000:
    sys.setrecursionlimit(10000)


class LaTeXCleaner:
    """
    Perfect LaTeX cleaner for 100% compliance
    """
    
    def __init__(self):
        self.formatting_commands = [
            r'\\centering',
            r'\\raggedright',
            r'\\raggedleft',
            r'\\noindent',
            r'\\vspace\*?\{[^}]*\}',
            r'\\hspace\*?\{[^}]*\}',
            r'\\vfill',
            r'\\hfill',
            r'\\newpage',
            r'\\clearpage',
            r'\\pagebreak',
            r'\\linebreak',
            r'\\smallskip',
            r'\\medskip',
            r'\\bigskip',
            r'\\par\b',
            r'\\maketitle',
            r'\\quad',
            r'\\qquad',
        ]
        
        self.table_formatting = [
            r'\[htpb!?\]',
            r'\[H\]',
            r'\[h!\]',
            r'\[t!\]',
            r'\[b!\]',
            r'\[p\]',
            r'\\midrule',
            r'\\toprule',
            r'\\bottomrule',
            r'\\hline',
            r'\\cline\{[^}]*\}',
        ]
        
        # FIX: ALL font/size commands
        self.font_commands = [
            # Font styles
            (r'\\textbf\{([^}]*)\}', r'\1'),
            (r'\\textit\{([^}]*)\}', r'\1'),
            (r'\\emph\{([^}]*)\}', r'\1'),
            (r'\\underline\{([^}]*)\}', r'\1'),
            (r'\\textrm\{([^}]*)\}', r'\1'),
            (r'\\textsf\{([^}]*)\}', r'\1'),
            (r'\\texttt\{([^}]*)\}', r'\1'),
            (r'\\text\{([^}]*)\}', r'\1'),
            
            # Font sizes
            (r'\\tiny\b', ''),
            (r'\\scriptsize\b', ''),
            (r'\\footnotesize\b', ''),
            (r'\\small\b', ''),
            (r'\\normalsize\b', ''),
            (r'\\large\b', ''),
            (r'\\Large\b', ''),
            (r'\\LARGE\b', ''),
            (r'\\huge\b', ''),
            (r'\\Huge\b', ''),
        ]
    
    def clean(self, latex_content: str) -> str:
        """Clean LaTeX for parsing"""
        if not latex_content:
            return ""
        
        text = latex_content
        text = self.remove_comments(text)
        
        for pattern in self.formatting_commands:
            text = re.sub(pattern, '', text)
        
        for pattern in self.table_formatting:
            text = re.sub(pattern, '', text)
        
        for pattern, replacement in self.font_commands:
            text = re.sub(pattern, replacement, text)
        
        text = self.normalize_inline_math(text)
        text = self.normalize_block_math(text)
        text = self.standardize_whitespace(text)
        
        return text.strip()
    
    def normalize_inline_math(self, text: str) -> str:
        """Normalize inline math"""
        text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)
        text = re.sub(
            r'\\begin\{math\}(.+?)\\end\{math\}',
            r'$\1$',
            text,
            flags=re.DOTALL
        )
        return text
    
    def normalize_block_math(self, text: str) -> str:
        """Normalize block math"""
        text = re.sub(
            r'\$\$(.+?)\$\$',
            r'\\begin{equation}\1\\end{equation}',
            text,
            flags=re.DOTALL
        )
        text = re.sub(
            r'\\\[(.+?)\\\]',
            r'\\begin{equation}\1\\end{equation}',
            text,
            flags=re.DOTALL
        )
        return text
    
    def remove_comments(self, text: str) -> str:
        """Remove LaTeX comments while preserving percentages"""
        lines = text.split('\n')
        result_lines = []
        
        for line in lines:
            ESCAPED = '\x00ESC\x00'
            protected = line.replace('\\%', ESCAPED)
            
            # Skip comment-only lines
            if re.match(r'^\s*%', protected):
                continue
            
            # Process inline comments
            result = []
            i = 0
            while i < len(protected):
                if protected[i:].startswith(ESCAPED):
                    result.append('%')
                    i += len(ESCAPED)
                elif protected[i] == '%':
                    # Check if percentage
                    if i > 0 and protected[i-1].isdigit():
                        result.append('%')
                        i += 1
                    else:
                        # Comment
                        break
                else:
                    result.append(protected[i])
                    i += 1
            
            result_str = ''.join(result).rstrip()
            result_lines.append(result_str)
        
        return '\n'.join(result_lines)
    
    def standardize_whitespace(self, text: str) -> str:
        """Standardize whitespace"""
        text = text.replace('\t', ' ')
        text = re.sub(r'[ ]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def to_plain_text(self, latex_content: str) -> str:
        """
        Convert to PERFECT plain text
        - Preserve citations as [key]
        - Remove ALL formatting
        - Remove ALL control characters
        """
        if not latex_content:
            return ""
        
        text = latex_content
        
        # Remove comments
        text = self.remove_comments(text)
        
        # Remove document structure
        structure_patterns = [
            r'\\begin\{abstract\}',
            r'\\end\{abstract\}',
            r'\\begin\{document\}',
            r'\\end\{document\}',
            r'\\begin\{tcolorbox\}.*?\]',
            r'\\end\{tcolorbox\}',
        ]
        for pattern in structure_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # Remove title/author metadata (keep content)
        text = re.sub(r'\\title\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\author\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\IEEEauthorblockN\{([^}]*)\}', r'\1', text)
        
        # Remove affiliation completely
        text = re.sub(r'\\affiliation\{[^}]*\}', '', text)
        text = re.sub(r'\\email\{[^}]*\}', '', text)
        text = re.sub(r'\\institution\{[^}]*\}', '', text)
        
        # Remove formatting
        for pattern in self.formatting_commands:
            text = re.sub(pattern, '', text)
        
        for pattern in self.table_formatting:
            text = re.sub(pattern, '', text)
        
        # FIX: Apply font commands RECURSIVELY
        # This handles nested \textbf{\textit{...}}
        max_iterations = 10
        for _ in range(max_iterations):
            prev = text
            for pattern, replacement in self.font_commands:
                text = re.sub(pattern, replacement, text)
            if prev == text:
                break
        
        # CRITICAL: Preserve citations as [key]
        # Handle ALL variants
        citation_pattern = r'~?\s*\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]+)\}'
        text = re.sub(citation_pattern, r'[\1]', text)
        
        # Also handle other citation commands
        text = re.sub(r'\\citeauthor\*?(?:\[[^\]]*\])?\{([^}]+)\}', r'[\1]', text)
        text = re.sub(r'\\citeyear\*?(?:\[[^\]]*\])?\{([^}]+)\}', r'[\1]', text)
        
        # Remove ref/label
        text = re.sub(r'\\ref\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\label\{[^}]*\}', '', text)
        
        # Convert special characters
        text = text.replace('~', ' ')
        text = text.replace('\\%', '%')
        text = text.replace('\\&', '&')
        text = text.replace('\\$', '$')
        text = text.replace('\\_', '_')
        
        # Remove positioning arguments
        text = re.sub(r'\[[htpb!]+\]', '', text)
        text = re.sub(r'\[width=[^\]]+\]', '', text)
        text = re.sub(r'\[height=[^\]]+\]', '', text)
        text = re.sub(r'\[scale=[^\]]+\]', '', text)
        
        # Remove image filenames
        text = re.sub(r'[\w_/-]+\.(png|jpg|jpeg|pdf|eps|svg)', '', text, flags=re.IGNORECASE)
        
        # Remove includegraphics
        text = re.sub(r'\\includegraphics\[[^\]]*\]\{[^}]*\}', '', text)
        text = re.sub(r'\\includegraphics\{[^}]*\}', '', text)
        
        # FIX: Remove remaining commands RECURSIVELY
        max_iterations = 10
        for _ in range(max_iterations):
            prev = text
            # Commands with arguments (keep content)
            text = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}', r'\1', text)
            if prev == text:
                break
        
        # Remove simple commands
        text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
        
        # Remove braces
        text = re.sub(r'[{}]', '', text)
        
        # Clean up line breaks
        text = re.sub(r'\\\\', ' ', text)
        
        # FIX: Remove ALL newlines and tabs
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('\r', ' ')
        
        # FIX: Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Normalize to single spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up punctuation spacing
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])([.,;:!?])', r'\1 \2', text)
        
        return text.strip()
    
    def clean_for_deduplication(self, text: str) -> str:
        """Clean for deduplication"""
        text = self.clean(text)
        
        text = text.lower()
        text = ' '.join(text.split())
        text = re.sub(r'[""\'\'`]', '"', text)
        text = re.sub(r'[—–−]', '-', text)
        
        return text.strip()