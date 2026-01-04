"""
Multi-file gathering for LaTeX sources
Handles \\input{}, \\include{}, and identifies main compilation file

This module identifies the main LaTeX compilation path and gathers only
files that are actually included in the final PDF rendering.
Unused or redundant files are ignored.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.file_io import read_tex_file


class FileGatherer:
    """
    Gather LaTeX files for processing
    
    Handles:
    - Finding main compilation file (contains \\documentclass and \\begin{document})
    - Recursively gathering \\input{} and \\include{} files
    - Processing multiple versions of a publication
    """
    
    def __init__(self, tex_dir):
        """
        Args:
            tex_dir: Path to tex folder containing all versions
        """
        self.tex_dir = Path(tex_dir)
        self.included_files = set()  # Track included files
        
        # Patterns for file inclusion
        self.input_patterns = [
            re.compile(r'\\input\{([^}]+)\}'),
            re.compile(r'\\include\{([^}]+)\}'),
            re.compile(r'\\subfile\{([^}]+)\}'),
            re.compile(r'\\import\{[^}]*\}\{([^}]+)\}'),
        ]
        
        # Main file indicators
        self.main_file_names = ['main.tex', 'paper.tex', 'article.tex', 'manuscript.tex']
        self.main_markers = [
            re.compile(r'\\documentclass'),
            re.compile(r'\\begin\{document\}'),
        ]
    
    def find_main_file(self, version_dir) -> Optional[Path]:
        """
        Find the main .tex file that compiles to PDF
        
        Heuristics (in order):
        1. File named main.tex, paper.tex, etc.
        2. File containing \\documentclass AND \\begin{document}
        3. File with most \\input/\\include commands
        
        Args:
            version_dir: Path to version folder
        
        Returns:
            Path to main .tex file, or None if not found
        """
        version_dir = Path(version_dir)
        
        if not version_dir.exists():
            return None
        
        tex_files = list(version_dir.glob('*.tex'))
        
        if not tex_files:
            # Check subdirectories
            tex_files = list(version_dir.glob('**/*.tex'))
        
        if not tex_files:
            return None
        
        # Strategy 1: Check for common main file names
        for tex_file in tex_files:
            if tex_file.name.lower() in self.main_file_names:
                # Verify it has documentclass
                try:
                    content = read_tex_file(tex_file)
                    if self._is_main_file(content):
                        return tex_file
                except Exception:
                    continue
        
        # Strategy 2: Find file with documentclass and begin{document}
        candidates = []
        for tex_file in tex_files:
            try:
                content = read_tex_file(tex_file)
                if self._is_main_file(content):
                    # Score by number of includes
                    include_count = self._count_includes(content)
                    candidates.append((tex_file, include_count))
            except Exception:
                continue
        
        if candidates:
            # Return the one with most includes (likely the main file)
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        # Strategy 3: Fallback - return first .tex file
        return tex_files[0] if tex_files else None
    
    def _is_main_file(self, content: str) -> bool:
        """Check if content indicates a main LaTeX file"""
        has_docclass = bool(re.search(r'\\documentclass', content))
        has_begin_doc = bool(re.search(r'\\begin\{document\}', content))
        return has_docclass and has_begin_doc
    
    def _count_includes(self, content: str) -> int:
        """Count number of \\input/\\include commands"""
        count = 0
        for pattern in self.input_patterns:
            count += len(pattern.findall(content))
        return count
    
    def gather_included_files(self, main_file) -> List[Path]:
        """
        Recursively gather all \\input{} and \\include{} files
        
        Args:
            main_file: Path to main .tex file
        
        Returns:
            List of all .tex files in compilation order
        """
        main_file = Path(main_file)
        self.included_files = set()
        
        files = self._gather_recursive(main_file)
        
        return files
    
    def _gather_recursive(self, tex_file, depth=0) -> List[Path]:
        """
        Recursively gather included files
        
        Args:
            tex_file: Current .tex file
            depth: Recursion depth (to prevent infinite loops)
        
        Returns:
            List of files in order
        """
        if depth > 10:  # Prevent infinite recursion
            return []
        
        tex_file = Path(tex_file)
        
        if not tex_file.exists():
            return []
        
        if tex_file in self.included_files:
            return []  # Already processed
        
        self.included_files.add(tex_file)
        
        files = [tex_file]
        
        try:
            content = read_tex_file(tex_file)
        except Exception:
            return files
        
        base_dir = tex_file.parent
        
        # Find all included files
        for pattern in self.input_patterns:
            for match in pattern.finditer(content):
                included_name = match.group(1)
                
                # Add .tex extension if not present
                if not included_name.endswith('.tex'):
                    included_name += '.tex'
                
                # Resolve path
                included_path = base_dir / included_name
                
                if included_path.exists():
                    # Recursively gather from included file
                    sub_files = self._gather_recursive(included_path, depth + 1)
                    files.extend(sub_files)
        
        return files
    
    def get_all_version_files(self) -> Dict[int, List[Path]]:
        """
        Get all files for all versions
        
        Returns:
            Dict: {version_num: [list of files]}
        """
        versions = {}
        
        # Look for version folders (e.g., 2504-13946v1, 2504-13946v2)
        if not self.tex_dir.exists():
            return versions
        
        # Find version directories
        version_pattern = re.compile(r'v(\d+)$')
        
        for item in sorted(self.tex_dir.iterdir()):
            if item.is_dir():
                # Try to extract version number
                match = version_pattern.search(item.name)
                
                if match:
                    version_num = int(match.group(1))
                else:
                    # Assume single version or numbered differently
                    version_num = 1
                
                main_file = self.find_main_file(item)
                
                if main_file:
                    files = self.gather_included_files(main_file)
                    versions[version_num] = files
        
        # If no version folders found, treat the whole directory as version 1
        if not versions:
            main_file = self.find_main_file(self.tex_dir)
            if main_file:
                files = self.gather_included_files(main_file)
                versions[1] = files
        
        return versions
    
    def get_ignored_files(self, version_dir) -> List[Path]:
        """
        Get list of .tex files that are NOT included in compilation
        
        Args:
            version_dir: Path to version folder
        
        Returns:
            List of ignored file paths
        """
        version_dir = Path(version_dir)
        
        main_file = self.find_main_file(version_dir)
        if not main_file:
            return []
        
        included_files = set(self.gather_included_files(main_file))
        
        all_tex_files = set(version_dir.glob('**/*.tex'))
        
        ignored = all_tex_files - included_files
        
        return list(ignored)
    
    def combine_files(self, files: List[Path]) -> str:
        """
        Combine content from multiple .tex files
        
        Resolves \input and \include commands by inserting file contents
        
        Args:
            files: List of file paths to combine
        
        Returns:
            Combined LaTeX content as single string
        """
        combined = ""
        
        for file_path in files:
            try:
                content = read_tex_file(file_path)
                combined += f"\n% === From {file_path.name} ===\n"
                combined += content + "\n"
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        
        return combined
    
    def get_file_statistics(self) -> Dict:
        """
        Get statistics about gathered files
        
        Returns:
            Dict with file statistics
        """
        versions = self.get_all_version_files()
        
        stats = {
            'total_versions': len(versions),
            'total_files': sum(len(files) for files in versions.values()),
            'files_per_version': {},
        }
        
        for version, files in versions.items():
            stats['files_per_version'][version] = len(files)
        
        return stats