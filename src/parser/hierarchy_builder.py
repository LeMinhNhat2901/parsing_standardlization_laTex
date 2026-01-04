"""
Build hierarchical tree structure from LaTeX
Handles: Chapters, Sections, Subsections, Paragraphs, 
         Sentences, Formulas, Figures, Tables, Itemize

As per requirement 2.1.2:
- Document acts as root
- Chapters/Sections comprise second level
- Subsections, Paragraphs form lower levels
- Leaf nodes: Sentences, Block Formulas, Figures (including Tables)
- Itemize blocks are higher components with each item being next-level
- Exclude References sections
- Include Acknowledgements and Appendices
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


class HierarchyBuilder:
    """
    Build hierarchical tree structure from LaTeX content
    """
    
    def __init__(self, arxiv_id: str):
        """
        Args:
            arxiv_id: e.g., "2504-13946"
        """
        self.arxiv_id = arxiv_id
        self.elements = {}  # {element_id: content}
        self.hierarchy = defaultdict(dict)  # {version: {child_id: parent_id}}
        self.id_counter = 0
        self.document_type = 'paper'  # 'paper' or 'book'
        
        # Section commands in order of hierarchy
        self.section_commands = [
            'chapter', 'section', 'subsection', 
            'subsubsection', 'paragraph', 'subparagraph'
        ]
        
        # Sections to exclude
        self.exclude_patterns = [
            r'references?', r'bibliography', r'bibliographie',
            r'r[eé]f[eé]rences?'
        ]
        
        # Sections to include even if unnumbered
        self.include_patterns = [
            r'acknowledge?ments?', r'appendix', r'appendices',
            r'supplementary', r'annex'
        ]
        
        # Block formula environments (leaf nodes)
        self.formula_environments = [
            'equation', 'equation*', 'align', 'align*',
            'eqnarray', 'eqnarray*', 'gather', 'gather*',
            'multline', 'multline*', 'displaymath'
        ]
    
    def parse_latex(self, latex_content: str, version: int):
        """
        Parse LaTeX content and build hierarchy
        
        Args:
            latex_content: Combined content from all files
            version: Version number (1, 2, 3, ...)
        """
        # Detect document type (book vs paper)
        self.document_type = self._detect_document_type(latex_content)
        
        # Create root document element
        root_id = self._generate_element_id('doc')
        self.elements[root_id] = f"Document - {self.arxiv_id}"
        
        # Extract document body
        body = self._extract_body(latex_content)
        
        # Parse structure recursively
        self._parse_structure(body, root_id, version, level=0)
    
    def _detect_document_type(self, content: str) -> str:
        """Detect if document is book/thesis or paper format"""
        if re.search(r'\\documentclass.*\{book\}', content):
            return 'book'
        if re.search(r'\\chapter\b', content):
            return 'book'
        return 'paper'
    
    def _extract_body(self, content: str) -> str:
        """Extract content between \\begin{document} and \\end{document}"""
        match = re.search(
            r'\\begin\{document\}(.*)\\end\{document\}',
            content,
            re.DOTALL
        )
        if match:
            return match.group(1)
        return content
    
    def _parse_structure(self, content: str, parent_id: str, version: int, level: int):
        """
        Recursively parse LaTeX structure
        
        Handles:
        - \\chapter, \\section, \\subsection, \\paragraph
        - \\begin{itemize}...\\item...\\end{itemize}
        - Sentences (split by .)
        - \\begin{equation}, $$...$$
        - \\begin{figure}, \\begin{table}
        """
        if level > 10:  # Prevent infinite recursion
            return
        
        # Find the next structural element
        current_pos = 0
        remaining_text = ""
        
        while current_pos < len(content):
            # Find next section, figure, table, equation, or itemize
            next_element = self._find_next_element(content, current_pos)
            
            if next_element is None:
                # No more structural elements, process remaining as text
                remaining_text += content[current_pos:]
                break
            
            elem_type, elem_start, elem_end, elem_content, elem_title = next_element
            
            # Process text before this element
            text_before = content[current_pos:elem_start].strip()
            if text_before:
                remaining_text += text_before + " "
            
            # Check if this section should be excluded
            if elem_type in self.section_commands and self._should_exclude(elem_title):
                current_pos = elem_end
                continue
            
            # Process accumulated text as sentences
            if remaining_text.strip():
                self._parse_text_content(remaining_text.strip(), parent_id, version)
                remaining_text = ""
            
            # Create element and process its content
            if elem_type in self.section_commands:
                # Section/chapter - higher component
                elem_id = self._generate_element_id(elem_type[:3])
                self.elements[elem_id] = elem_title  # Store title only
                self.hierarchy[version][elem_id] = parent_id
                
                # Recursively parse section content
                self._parse_structure(elem_content, elem_id, version, level + 1)
            
            elif elem_type == 'itemize' or elem_type == 'enumerate':
                # Itemize block - branching structure
                self._parse_itemize(elem_content, elem_type, parent_id, version)
            
            elif elem_type == 'figure':
                # Figure (leaf element)
                self._parse_figure_or_table(elem_content, 'figure', parent_id, version)
            
            elif elem_type == 'table':
                # Table (treated as Figure per requirements)
                self._parse_figure_or_table(elem_content, 'table-figure', parent_id, version)
            
            elif elem_type == 'equation':
                # Block formula (leaf element)
                elem_id = self._generate_element_id('formula')
                self.elements[elem_id] = elem_content
                self.hierarchy[version][elem_id] = parent_id
            
            current_pos = elem_end
        
        # Process any remaining text
        if remaining_text.strip():
            self._parse_text_content(remaining_text.strip(), parent_id, version)
    
    def _find_next_element(self, content: str, start_pos: int):
        """
        Find the next structural element in content
        
        Returns:
            Tuple (type, start, end, content, title) or None
        """
        elements_found = []
        
        # Look for sections (numbered and unnumbered)
        for cmd in self.section_commands:
            patterns = [
                # Numbered: \section{Title}
                re.compile(r'\\' + cmd + r'\{([^}]+)\}'),
                # Unnumbered: \section*{Title}
                re.compile(r'\\' + cmd + r'\*\{([^}]+)\}'),
            ]
            
            for pattern in patterns:
                for match in pattern.finditer(content, start_pos):
                    title = match.group(1)
                    section_start = match.start()
                    
                    # Find end of section (next section of same or higher level, or end)
                    section_end, section_content = self._find_section_end(
                        content, match.end(), cmd
                    )
                    
                    elements_found.append((
                        cmd, section_start, section_end, section_content, title
                    ))
                    break  # Only first match
        
        # Look for environments
        env_patterns = [
            ('itemize', r'\\begin\{itemize\}(.*?)\\end\{itemize\}'),
            ('enumerate', r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}'),
            ('figure', r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}'),
            ('table', r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}'),
        ]
        
        for env_type, pattern in env_patterns:
            match = re.search(pattern, content[start_pos:], re.DOTALL)
            if match:
                abs_start = start_pos + match.start()
                abs_end = start_pos + match.end()
                elements_found.append((
                    env_type, abs_start, abs_end, match.group(1), ""
                ))
        
        # Look for block equations
        for env in self.formula_environments:
            pattern = re.compile(
                r'\\begin\{' + env + r'\}(.*?)\\end\{' + env + r'\}',
                re.DOTALL
            )
            match = pattern.search(content, start_pos)
            if match:
                elements_found.append((
                    'equation', match.start(), match.end(), match.group(0), ""
                ))
        
        # Also check for $$...$$ that wasn't normalized
        match = re.search(r'\$\$(.+?)\$\$', content[start_pos:], re.DOTALL)
        if match:
            abs_start = start_pos + match.start()
            abs_end = start_pos + match.end()
            elements_found.append((
                'equation', abs_start, abs_end, match.group(0), ""
            ))
        
        # Return earliest element
        if elements_found:
            return min(elements_found, key=lambda x: x[1])
        
        return None
    
    def _find_section_end(self, content: str, start: int, current_cmd: str):
        """Find where a section ends (next same/higher level section or end)"""
        # Get level of current section
        try:
            current_level = self.section_commands.index(current_cmd)
        except ValueError:
            current_level = len(self.section_commands)
        
        # Find next section of same or higher level
        pattern = r'\\(' + '|'.join(self.section_commands[:current_level+1]) + r')\*?\{'
        
        match = re.search(pattern, content[start:])
        
        if match:
            end = start + match.start()
        else:
            end = len(content)
        
        section_content = content[start:end]
        
        return end, section_content
    
    def _parse_itemize(self, content: str, env_type: str, parent_id: str, version: int):
        """
        Parse itemize/enumerate as branching structure
        
        Structure:
            itemize-block (higher component)
            ├── item-1 (next-level element)
            ├── item-2 (next-level element)
            └── item-3 (next-level element)
        """
        # Create itemize block element
        block_id = self._generate_element_id(env_type)
        self.elements[block_id] = f"{env_type.capitalize()} block"
        self.hierarchy[version][block_id] = parent_id
        
        # Split by \item
        items = re.split(r'\\item\b', content)
        
        for i, item_content in enumerate(items[1:], 1):  # Skip first empty part
            item_content = item_content.strip()
            if not item_content:
                continue
            
            item_id = self._generate_element_id('item')
            
            # Check if item contains nested structures
            if '\\begin{itemize}' in item_content or '\\begin{enumerate}' in item_content:
                # Has nested list
                self.elements[item_id] = f"Item {i}"
                self.hierarchy[version][item_id] = block_id
                self._parse_structure(item_content, item_id, version, level=5)
            else:
                # Simple item - store content
                self.elements[item_id] = item_content
                self.hierarchy[version][item_id] = block_id
    
    def _parse_text_content(self, text: str, parent_id: str, version: int):
        """
        Parse text into sentences (leaf elements)
        
        Sentences are separated by periods, question marks, or exclamation marks
        """
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 3:  # Skip very short fragments
                continue
            
            sent_id = self._generate_element_id('sent')
            self.elements[sent_id] = sent
            self.hierarchy[version][sent_id] = parent_id
    
    def _parse_figure_or_table(self, content: str, elem_type: str, parent_id: str, version: int):
        """
        Extract figure/table as leaf element
        Note: Tables are treated as a type of Figure
        """
        elem_id = self._generate_element_id(elem_type.split('-')[0])
        
        # Extract caption if exists
        caption_match = re.search(r'\\caption\{([^}]+)\}', content)
        caption = caption_match.group(1) if caption_match else ""
        
        # Extract label if exists
        label_match = re.search(r'\\label\{([^}]+)\}', content)
        label = label_match.group(1) if label_match else ""
        
        self.elements[elem_id] = {
            'type': elem_type,
            'caption': caption,
            'label': label,
            'content': content
        }
        
        self.hierarchy[version][elem_id] = parent_id
    
    def _should_exclude(self, section_title: str) -> bool:
        """
        Check if section should be excluded (e.g., References)
        
        Returns:
            True if should exclude, False otherwise
        """
        title_lower = section_title.lower().strip()
        
        # Check exclusion patterns
        for pattern in self.exclude_patterns:
            if re.search(pattern, title_lower):
                return True
        
        return False
    
    def _should_include(self, section_title: str) -> bool:
        """
        Check if section should be included (even if unnumbered)
        
        Returns:
            True if should include, False otherwise
        """
        title_lower = section_title.lower().strip()
        
        for pattern in self.include_patterns:
            if re.search(pattern, title_lower):
                return True
        
        return False
    
    def _generate_element_id(self, element_type: str) -> str:
        """
        Generate unique ID for element
        
        Format: {publication_id}-{element_type}-{counter}
        
        Examples:
            "2504-13946-doc-1"
            "2504-13946-sec-2"
            "2504-13946-sent-42"
        """
        self.id_counter += 1
        return f"{self.arxiv_id}-{element_type}-{self.id_counter}"
    
    def get_hierarchy_json(self) -> Dict:
        """
        Export hierarchy in required JSON format
        
        Format:
        {
            "elements": {
                "smallest-element-id": "Cleaned latex content...",
                "higher-component-id": "Title of the associated content"
            },
            "hierarchy": {
                "1": {  // Version 1
                    "higher-component-id": "root-document-id",
                    "smallest-element-id": "higher-component-id"
                },
                "2": {  // Version 2
                    "id-string-x": "id-string-y"
                }
            }
        }
        
        Returns:
            Dict with 'elements' and 'hierarchy' keys
        """
        # Convert elements (handle dict values for figures/tables)
        elements_output = {}
        for elem_id, content in self.elements.items():
            if isinstance(content, dict):
                # For figures/tables, just store caption
                elements_output[elem_id] = content.get('caption', str(content))
            else:
                elements_output[elem_id] = content
        
        # Convert hierarchy (defaultdict to regular dict, version keys to strings)
        hierarchy_output = {}
        for version, mappings in self.hierarchy.items():
            hierarchy_output[str(version)] = dict(mappings)
        
        return {
            'elements': elements_output,
            'hierarchy': hierarchy_output
        }
    
    def get_statistics(self) -> Dict:
        """Get parsing statistics"""
        stats = {
            'total_elements': len(self.elements),
            'versions': len(self.hierarchy),
            'sentences': 0,
            'formulas': 0,
            'figures': 0,
            'sections': 0,
            'items': 0,
        }
        
        for elem_id in self.elements.keys():
            if '-sent-' in elem_id:
                stats['sentences'] += 1
            elif '-formula-' in elem_id:
                stats['formulas'] += 1
            elif '-fig-' in elem_id or '-table-' in elem_id:
                stats['figures'] += 1
            elif '-sec-' in elem_id or '-subsec-' in elem_id:
                stats['sections'] += 1
            elif '-item-' in elem_id:
                stats['items'] += 1
        
        return stats