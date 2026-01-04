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
from typing import List


class LaTeXCleaner:
    """
    Clean and normalize LaTeX content for hierarchy construction
    """
    
    def __init__(self):
        # Formatting commands to remove (don't contribute to semantic meaning)
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
        ]
        
        # Table/figure formatting to remove
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
            r'\\cmidrule\{[^}]*\}',
            r'\\multicolumn\{[^}]*\}\{[^}]*\}',
            r'\\multirow\{[^}]*\}\{[^}]*\}',
        ]
        
        # Font commands to simplify
        self.font_commands = [
            (r'\\textbf\{([^}]*)\}', r'\1'),
            (r'\\textit\{([^}]*)\}', r'\1'),
            (r'\\emph\{([^}]*)\}', r'\1'),
            (r'\\underline\{([^}]*)\}', r'\1'),
            (r'\\textrm\{([^}]*)\}', r'\1'),
            (r'\\textsf\{([^}]*)\}', r'\1'),
            (r'\\texttt\{([^}]*)\}', r'\1'),
            (r'\\text\{([^}]*)\}', r'\1'),
            (r'\\mathrm\{([^}]*)\}', r'\1'),
            (r'\\mathbf\{([^}]*)\}', r'\1'),
            (r'\\mathit\{([^}]*)\}', r'\1'),
        ]
    
    def clean(self, latex_content: str) -> str:
        """
        Clean LaTeX content by removing non-semantic formatting
        
        Args:
            latex_content: Raw LaTeX string
        
        Returns:
            Cleaned LaTeX string
        """
        if not latex_content:
            return ""
        
        text = latex_content
        
        # Step 1: Remove comments
        text = self.remove_comments(text)
        
        # Step 2: Remove formatting commands
        for pattern in self.formatting_commands:
            text = re.sub(pattern, '', text)
        
        # Step 3: Remove table formatting
        for pattern in self.table_formatting:
            text = re.sub(pattern, '', text)
        
        # Step 4: Simplify font commands (keep content)
        for pattern, replacement in self.font_commands:
            text = re.sub(pattern, replacement, text)
        
        # Step 5: Normalize math
        text = self.normalize_inline_math(text)
        text = self.normalize_block_math(text)
        
        # Step 6: Standardize whitespace
        text = self.standardize_whitespace(text)
        
        return text.strip()
    
    def normalize_inline_math(self, text: str) -> str:
        """
        Normalize inline math to $...$
        
        Converts:
        - \\(...\\) → $...$
        - \\begin{math}...\\end{math} → $...$
        
        Args:
            text: LaTeX text
        
        Returns:
            Text with normalized inline math
        """
        # Convert \(...\) to $...$
        text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)
        
        # Convert \begin{math}...\end{math} to $...$
        text = re.sub(
            r'\\begin\{math\}(.+?)\\end\{math\}',
            r'$\1$',
            text,
            flags=re.DOTALL
        )
        
        return text
    
    def normalize_block_math(self, text: str) -> str:
        """
        Normalize block math to \\begin{equation}...\\end{equation}
        
        Converts:
        - $$...$$ → \\begin{equation}...\\end{equation}
        - \\[...\\] → \\begin{equation}...\\end{equation}
        - \\begin{displaymath}...\\end{displaymath} → \\begin{equation}...\\end{equation}
        
        Note: Other equation environments (align, eqnarray, etc.) are kept as-is
        since they have special semantics (alignment, multiple equations)
        
        Args:
            text: LaTeX text
        
        Returns:
            Text with normalized block math
        """
        # Convert $$...$$ to \begin{equation}...\end{equation}
        text = re.sub(
            r'\$\$(.+?)\$\$',
            r'\\begin{equation}\1\\end{equation}',
            text,
            flags=re.DOTALL
        )
        
        # Convert \[...\] to \begin{equation}...\end{equation}
        text = re.sub(
            r'\\\[(.+?)\\\]',
            r'\\begin{equation}\1\\end{equation}',
            text,
            flags=re.DOTALL
        )
        
        # Convert displaymath to equation
        text = re.sub(
            r'\\begin\{displaymath\}(.+?)\\end\{displaymath\}',
            r'\\begin{equation}\1\\end{equation}',
            text,
            flags=re.DOTALL
        )
        
        return text
    
    def remove_comments(self, text: str) -> str:
        """
        Remove LaTeX comments (lines starting with % or inline %)
        
        Note: Preserves \\% (escaped percent sign)
        
        Args:
            text: LaTeX text
        
        Returns:
            Text without comments
        """
        # Remove inline comments (but not escaped \%)
        # First, temporarily replace \% with placeholder
        text = text.replace('\\%', '%%ESCAPED_PERCENT%%')
        
        # Remove comments
        text = re.sub(r'%[^\n]*', '', text)
        
        # Restore escaped percent
        text = text.replace('%%ESCAPED_PERCENT%%', '\\%')
        
        return text
    
    def standardize_whitespace(self, text: str) -> str:
        """
        Remove excessive whitespace while preserving structure
        
        Args:
            text: LaTeX text
        
        Returns:
            Text with standardized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing whitespace on each line
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def extract_text_only(self, latex_content: str) -> str:
        """
        Extract plain text from LaTeX, removing all commands
        
        Args:
            latex_content: LaTeX content
        
        Returns:
            Plain text
        """
        text = self.clean(latex_content)
        
        # Remove remaining LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
        
        # Remove remaining braces
        text = re.sub(r'[{}]', '', text)
        
        # Remove math
        text = re.sub(r'\$[^$]+\$', '', text)
        text = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', '', text, flags=re.DOTALL)
        
        # Clean up
        text = self.standardize_whitespace(text)
        
        return text.strip()
    
    def clean_for_deduplication(self, text: str) -> str:
        """
        Clean text specifically for content deduplication comparison
        
        This removes all formatting variations that might cause
        identical content to appear different.
        
        Args:
            text: Text to clean
        
        Returns:
            Normalized text for comparison
        """
        # First apply normal cleaning
        text = self.clean(text)
        
        # Additional normalization for comparison
        # Lowercase
        text = text.lower()
        
        # Remove all extra whitespace
        text = ' '.join(text.split())
        
        # Normalize quotes
        text = re.sub(r'[""\'\'`]', '"', text)
        
        # Normalize dashes
        text = re.sub(r'[–—−]', '-', text)
        
        return text.strip()