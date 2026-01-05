"""
Main script to run LaTeX parsing pipeline

This script implements the complete parsing workflow as per requirement 2.1:
1. Gather LaTeX files (handle \\input/\\include)
2. Build hierarchical structure
3. Clean/standardize content
4. Extract references
5. Deduplicate content and references
6. Save outputs (hierarchy.json, refs.bib)

Usage:
    python main_parser.py --input-dir /path/to/arxiv/source --output-dir /path/to/output
    python main_parser.py --input-dir /data/2304.12345 --arxiv-id 2304.12345
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from parser.file_gatherer import FileGatherer
from parser.latex_cleaner import LaTeXCleaner
from parser.reference_extractor import ReferenceExtractor
from parser.hierarchy_builder import HierarchyBuilder
from parser.deduplicator import Deduplicator
from utils.file_io import save_json, load_json, save_bibtex
from utils.logger import setup_logger, ProgressLogger

import config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Parse LaTeX source files into hierarchical structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Parse a single publication
    python main_parser.py --input-dir ./data/2304.12345 --output-dir ./output
    
    # Parse with explicit arXiv ID
    python main_parser.py --input-dir ./data/2304.12345 --arxiv-id 2304.12345
    
    # Process multiple publications
    python main_parser.py --input-dir ./data --batch --output-dir ./output
        """
    )
    
    parser.add_argument('--input-dir', '-i', required=True,
                       help='Input directory containing LaTeX source files')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory (default: from config or same as input)')
    parser.add_argument('--arxiv-id', 
                       help='ArXiv ID (extracted from path if not provided)')
    parser.add_argument('--batch', action='store_true',
                       help='Process all subdirectories as separate publications')
    parser.add_argument('--clean-only', action='store_true',
                       help='Only clean LaTeX (skip hierarchy building)')
    parser.add_argument('--refs-only', action='store_true',
                       help='Only extract references')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-dedup', action='store_true',
                       help='Skip deduplication step')
    
    return parser.parse_args()


def extract_arxiv_id(input_dir: str) -> str:
    """Extract arXiv ID from directory path"""
    # Try to extract from directory name
    dir_name = Path(input_dir).name
    
    # Common patterns: 2304.12345, 1606.03490, etc.
    import re
    pattern = r'\d{4}\.\d{4,5}'
    match = re.search(pattern, dir_name)
    
    if match:
        return match.group()
    
    return dir_name  # Use directory name as fallback


def process_publication(input_dir: str, output_dir: str, arxiv_id: str,
                       logger, args) -> dict:
    """
    Process a single publication
    
    Returns:
        Dict with statistics and output paths
    """
    logger.info(f"Processing publication: {arxiv_id}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        'arxiv_id': arxiv_id,
        'input_dir': input_dir,
        'output_dir': output_dir,
        'files_found': 0,
        'versions': 0,
        'elements': 0,
        'references': 0,
        'deduplicated_refs': 0
    }
    
    progress = ProgressLogger(6, "Parsing Pipeline")
    
    # ========================================
    # Step 1: Gather files
    # ========================================
    progress.start_step(1, "Gathering files")
    
    gatherer = FileGatherer(input_dir)
    version_files = gatherer.get_all_version_files()
    
    if not version_files:
        logger.warning(f"No LaTeX files found in {input_dir}")
        return stats
    
    stats['versions'] = len(version_files)
    stats['files_found'] = sum(len(files) for files in version_files.values())
    
    logger.info(f"Found {stats['files_found']} files across {stats['versions']} version(s)")
    progress.complete_step(1)
    
    # ========================================
    # Step 2: Build hierarchy
    # ========================================
    progress.start_step(2, "Building hierarchy")
    
    builder = HierarchyBuilder(arxiv_id)
    cleaner = LaTeXCleaner()
    
    for version, files in version_files.items():
        logger.info(f"Processing version: {version} ({len(files)} files)")
        
        # Combine all files for this version
        combined_content = gatherer.combine_files(files)
        
        # Pre-clean content (normalize whitespace, etc.)
        cleaned_content = cleaner.clean(combined_content)
        
        # Parse into hierarchy
        builder.parse_latex(cleaned_content, version)
    
    stats['elements'] = len(builder.get_elements())
    logger.info(f"Built hierarchy with {stats['elements']} elements")
    progress.complete_step(2)
    
    # ========================================
    # Step 3: Clean content
    # ========================================
    progress.start_step(3, "Cleaning content")
    
    # Clean all element content
    elements = builder.get_elements()
    for elem_id, elem in elements.items():
        # Elements can be strings or dicts
        if isinstance(elem, dict):
            if 'content' in elem:
                elem['content'] = cleaner.clean(elem['content'])
            if 'text' in elem:
                elem['text'] = cleaner.clean(elem['text'])
        elif isinstance(elem, str):
            # Element is a string - clean it directly
            elements[elem_id] = cleaner.clean(elem)
    
    logger.info("Cleaned all element content")
    progress.complete_step(3)
    
    # ========================================
    # Step 4: Extract references
    # ========================================
    progress.start_step(4, "Extracting references")
    
    ref_extractor = ReferenceExtractor()
    
    # Get all latex files content
    all_content = ""
    for files in version_files.values():
        all_content += gatherer.combine_files(files) + "\n"
    
    # Extract bibitem entries
    bibtex_entries = ref_extractor.extract_bibitems(all_content)
    
    # Try to load existing .bib files
    for files in version_files.values():
        for file in files:
            # file is a Path object, check suffix
            if isinstance(file, Path) and file.suffix == '.bib':
                file_entries = ref_extractor.parse_bib_file(str(file))
                bibtex_entries.update(file_entries)
            elif isinstance(file, str) and file.endswith('.bib'):
                file_entries = ref_extractor.parse_bib_file(file)
                bibtex_entries.update(file_entries)
    
    stats['references'] = len(bibtex_entries)
    logger.info(f"Extracted {stats['references']} references")
    progress.complete_step(4)
    
    # ========================================
    # Step 5: Deduplicate
    # ========================================
    if not args.no_dedup:
        progress.start_step(5, "Deduplicating")
        
        deduper = Deduplicator()
        
        # 5a. Deduplicate references (with citation renaming)
        all_latex_files = []
        for files in version_files.values():
            all_latex_files.extend(files)
        
        deduplicated_refs = deduper.deduplicate_references(
            bibtex_entries,
            all_latex_files
        )
        
        # 5b. Deduplicate content across versions
        # Build elements_dict: handle both string and dict elements
        elements_dict = {}
        for elem_id, elem in elements.items():
            if isinstance(elem, dict):
                # Dict element - extract content/text
                content = elem.get('content', elem.get('text', ''))
            else:
                # String element - use directly
                content = str(elem)
            elements_dict[elem_id] = content
        
        deduped_elements, id_mapping = deduper.deduplicate_content(elements_dict)
        
        stats['deduplicated_refs'] = len(deduplicated_refs)
        bibtex_entries = deduplicated_refs
        
        rename_map = deduper.get_rename_map()
        if rename_map:
            logger.info(f"Renamed {len(rename_map)} duplicate citations")
        
        progress.complete_step(5)
    else:
        progress.skip_step(5, "Deduplication (skipped)")
    
    # ========================================
    # Step 6: Save outputs
    # ========================================
    progress.start_step(6, "Saving outputs")
    
    # Save hierarchy.json
    hierarchy_path = os.path.join(output_dir, 'hierarchy.json')
    hierarchy_data = builder.get_hierarchy_json()
    save_json(hierarchy_path, hierarchy_data)
    logger.info(f"Saved hierarchy to {hierarchy_path}")
    
    # Save refs.bib
    refs_path = os.path.join(output_dir, 'refs.bib')
    save_bibtex(refs_path, bibtex_entries)
    logger.info(f"Saved {len(bibtex_entries)} references to {refs_path}")
    
    # Copy metadata.json and references.json from input (Lab 1 data)
    # These files are required by the matcher but come from Lab 1 crawling
    import shutil
    for filename in ['metadata.json', 'references.json']:
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            logger.info(f"Copied {filename} from input directory")
        else:
            logger.warning(f"Warning: {filename} not found in input directory - required for matching!")
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'parsing_stats.json')
    save_json(stats_path, stats)
    
    progress.complete_step(6)
    
    logger.info(f"✅ Parsing complete for {arxiv_id}")
    logger.info(f"   Elements: {stats['elements']}")
    logger.info(f"   References: {stats['references']}")
    
    return stats


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger('parser', level=log_level)
    
    # Determine output directory
    if args.output_dir:
        output_base = args.output_dir
    elif hasattr(config, 'OUTPUT_DIR') and config.OUTPUT_DIR:
        output_base = config.OUTPUT_DIR
    else:
        output_base = args.input_dir
    
    if args.batch:
        # Process all subdirectories
        input_path = Path(args.input_dir)
        all_stats = []
        
        for subdir in input_path.iterdir():
            if not subdir.is_dir():
                continue
            
            arxiv_id = extract_arxiv_id(str(subdir))
            output_dir = os.path.join(output_base, arxiv_id)
            
            try:
                stats = process_publication(
                    str(subdir), output_dir, arxiv_id, logger, args
                )
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Error processing {arxiv_id}: {e}")
                continue
        
        # Save overall statistics
        summary_path = os.path.join(output_base, 'parsing_summary.json')
        save_json(summary_path, {
            'total_publications': len(all_stats),
            'publications': all_stats
        })
        
        logger.info(f"✅ Batch processing complete: {len(all_stats)} publications")
    
    else:
        # Process single publication
        arxiv_id = args.arxiv_id or extract_arxiv_id(args.input_dir)
        output_dir = os.path.join(output_base, arxiv_id) if output_base != args.input_dir else output_base
        
        stats = process_publication(
            args.input_dir, output_dir, arxiv_id, logger, args
        )
        
        print(f"\n{'='*60}")
        print(f"PARSING SUMMARY")
        print(f"{'='*60}")
        print(f"ArXiv ID:    {stats['arxiv_id']}")
        print(f"Versions:    {stats['versions']}")
        print(f"Files:       {stats['files_found']}")
        print(f"Elements:    {stats['elements']}")
        print(f"References:  {stats['references']}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()