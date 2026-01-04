"""
Multi-file gathering for LaTeX sources
Handles \input{}, \include{}, and identifies main compilation file
"""

class FileGatherer:
    def __init__(self, tex_dir):
        """
        Args:
            tex_dir: Path to tex folder containing all versions
        """
        pass
    
    def find_main_file(self, version_dir):
        """
        Find the main .tex file that compiles to PDF
        Heuristics:
        - Contains \documentclass
        - Contains \begin{document}
        - Usually named main.tex, paper.tex, or arxiv_id.tex
        
        Returns:
            Path to main .tex file
        """
        pass
    
    def gather_included_files(self, main_file):
        """
        Recursively gather all \input{} and \include{} files
        
        Returns:
            List of all .tex files in compilation order
        """
        pass
    
    def get_all_version_files(self):
        """
        Get all files for all versions
        
        Returns:
            Dict: {version_num: [list of files]}
        """
        pass