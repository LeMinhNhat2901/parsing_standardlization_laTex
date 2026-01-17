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

# CRITICAL: Increase recursion limit BEFORE any library imports
# This prevents RecursionError with complex LaTeX files
sys.setrecursionlimit(10000)

# Disable pyparsing packrat to prevent recursion issues with deeply nested LaTeX
try:
    import pyparsing
    pyparsing.ParserElement.disablePackrat()
except (ImportError, AttributeError):
    pass  # pyparsing not installed or doesn't have this method

sys.path.insert(0, str(Path(__file__).parent))

from parser.file_gatherer import FileGatherer
from parser.latex_cleaner import LaTeXCleaner
from parser.reference_extractor import ReferenceExtractor
from parser.hierarchy_builder import HierarchyBuilder
from parser.deduplicator import Deduplicator
from utils.file_io import save_json, save_bibtex
from utils.logger import setup_logger


def parse_publication(input_dir: str, output_dir: str, arxiv_id: str, logger, args):
    """
    Process a single publication with COMPLETE reference extraction
    
    Returns:
        Dict with statistics and output paths
    """
    logger.info(f"Processing publication: {arxiv_id}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    
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
    
    # ========================================
    # CRITICAL: Copy metadata.json and references.json
    # ========================================
    import shutil
    for filename in ['metadata.json', 'references.json']:
        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"Copied {filename}")
        else:
            logger.warning(f"Warning: {filename} not found")
    
    # ========================================
    # Step 1: Gather files
    # ========================================
    logger.info("\n=== Step 1: Gathering files ===")
    
    gatherer = FileGatherer(input_dir)
    version_files = gatherer.get_all_version_files()
    
    if not version_files:
        logger.warning(f"No LaTeX files found")
        # Save empty outputs
        save_json(os.path.join(output_dir, 'hierarchy.json'), {'elements': {}, 'hierarchy': {}})
        
        # IMPORTANT: Still try to extract references even without LaTeX
        ref_extractor = ReferenceExtractor()
        all_refs = ref_extractor.extract_all_from_directory(Path(input_dir))
        ref_extractor.save_to_bib_file(Path(output_dir) / 'refs.bib')
        
        save_json(os.path.join(output_dir, 'parsing_stats.json'), stats)
        return stats
    
    stats['versions'] = len(version_files)
    stats['files_found'] = sum(len(files) for files in version_files.values())
    
    logger.info(f"Found {stats['files_found']} files across {stats['versions']} version(s)")
    
    # ========================================
    # Step 2: Build hierarchy
    # ========================================
    logger.info("\n=== Step 2: Building hierarchy ===")
    
    builder = HierarchyBuilder(arxiv_id)
    cleaner = LaTeXCleaner()
    
    for version, files in version_files.items():
        logger.info(f"Processing version {version}: {len(files)} files")
        
        # Combine files
        combined_content = gatherer.combine_files(files)
        
        # Light clean (keep structure for parsing)
        light_cleaned = cleaner.remove_comments(combined_content)
        
        # Parse hierarchy
        builder.parse_latex(light_cleaned, version)
    
    stats['elements'] = len(builder.get_elements())
    logger.info(f"Built hierarchy: {stats['elements']} elements")
    
    # ========================================
    # Step 3: Clean content
    # ========================================
    logger.info("\n=== Step 3: Cleaning content ===")
    
    elements = builder.get_elements()
    for elem_id, elem in elements.items():
        # Use type() instead of isinstance() to avoid recursion issues
        if type(elem).__name__ == 'dict':
            if 'content' in elem:
                elem['content'] = cleaner.clean(elem['content'])
            if 'text' in elem:
                elem['text'] = cleaner.clean(elem['text'])
        elif type(elem).__name__ == 'str':
            elements[elem_id] = cleaner.clean(elem)
    
    logger.info("Cleaned all elements")
    
    # ========================================
    # Step 4: Extract references (CRITICAL)
    # ========================================
    logger.info("\n=== Step 4: Extracting references ===")
    
    ref_extractor = ReferenceExtractor()
    
    # CRITICAL: Extract from ALL version directories
    all_refs = {}
    
    for version, files in version_files.items():
        # Get directory of first file
        if files:
            version_dir = files[0].parent
            logger.info(f"Extracting references from version {version}: {version_dir}")
            
            # Extract from this version directory
            version_refs = ref_extractor.extract_all_from_directory(version_dir)
            all_refs.update(version_refs)
    
    # Also check root input directory
    logger.info(f"Extracting references from root: {input_dir}")
    root_refs = ref_extractor.extract_all_from_directory(Path(input_dir))
    all_refs.update(root_refs)
    
    # Update extractor's entries
    ref_extractor.bibtex_entries = all_refs
    
    stats['references'] = len(all_refs)
    logger.info(f"Extracted {stats['references']} references")
    
    # DEBUG: Log reference count for verification
    logger.info(f"📊 References after Step 4: {len(all_refs)} entries")
    
    if stats['references'] == 0:
        logger.warning("⚠️  WARNING: No references found! Check:")
        logger.warning("  - Are there .bib files in the directory?")
        logger.warning("  - Are there \\bibitem entries in .tex files?")
    
    # ========================================
    # Step 5: Deduplicate
    # ========================================
    refs_before_dedup = len(all_refs)
    if not args.no_dedup and all_refs:
        logger.info("\n=== Step 5: Deduplicating ===")
        logger.info(f"   References before deduplication: {refs_before_dedup}")
        
        deduper = Deduplicator()
        
        # Get all latex files
        all_latex_files = []
        for files in version_files.values():
            all_latex_files.extend(files)
        
        # Deduplicate references
        deduplicated_refs = deduper.deduplicate_references(
            all_refs,
            all_latex_files
        )
        
        stats['deduplicated_refs'] = len(deduplicated_refs)
        
        # SAFETY CHECK: Ensure deduplication didn't remove too many references
        if len(deduplicated_refs) == 0 and refs_before_dedup > 0:
            logger.warning(f"⚠️  WARNING: Deduplication removed ALL references! Keeping original {refs_before_dedup} refs.")
            # Keep original references if deduplication fails
            stats['deduplicated_refs'] = refs_before_dedup
        else:
            all_refs = deduplicated_refs
            logger.info(f"   References after deduplication: {len(all_refs)}")
        
        rename_map = deduper.get_rename_map()
        if rename_map:
            logger.info(f"Renamed {len(rename_map)} duplicate citations")
        
        # Deduplicate content
        elements_dict = {}
        for elem_id, elem in elements.items():
            # Use type() instead of isinstance() to avoid recursion issues
            if type(elem).__name__ == 'dict':
                content = elem.get('content', elem.get('text', ''))
            else:
                content = str(elem)
            elements_dict[elem_id] = content
        
        deduped_elements, id_mapping = deduper.deduplicate_content(elements_dict)
        
        builder.elements = deduped_elements
        
        # Update hierarchy with canonical IDs
        updated_hierarchy = builder.get_hierarchy()
        for version, mappings in updated_hierarchy.items():
            new_mappings = {}
            for child_id, parent_id in mappings.items():
                canonical_child = id_mapping.get(child_id, child_id)
                canonical_parent = id_mapping.get(parent_id, parent_id)
                
                if canonical_child in deduped_elements:
                    new_mappings[canonical_child] = canonical_parent
            
            updated_hierarchy[version] = new_mappings
        
        builder.hierarchy = updated_hierarchy
        
        logger.info(f"Deduplicated: {len(elements_dict)} → {len(deduped_elements)} elements")
    else:
        if not all_refs:
            logger.warning("Skipping deduplication (no references)")
    
    # ========================================
    # Step 6: Save outputs
    # ========================================
    logger.info("\n=== Step 6: Saving outputs ===")
    
    # Save hierarchy.json
    hierarchy_path = os.path.join(output_dir, 'hierarchy.json')
    hierarchy_data = builder.get_hierarchy_json()
    save_json(hierarchy_path, hierarchy_data)
    logger.info(f"Saved hierarchy: {hierarchy_path}")
    
    # Save refs.bib (CRITICAL)
    refs_path = os.path.join(output_dir, 'refs.bib')
    if all_refs:
        save_bibtex(refs_path, all_refs)
        logger.info(f"✅ Saved {len(all_refs)} references to {refs_path}")
    else:
        # Create empty file
        Path(refs_path).write_text("", encoding='utf-8')
        logger.warning(f"⚠️  Created empty refs.bib (no references found)")
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'parsing_stats.json')
    save_json(stats_path, stats)
    
    logger.info(f"\n✅ Parsing complete for {arxiv_id}")
    logger.info(f"   Elements: {stats['elements']}")
    logger.info(f"   References: {stats['references']}")
    
    return stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Parse LaTeX with reference extraction')
    
    parser.add_argument('--input-dir', '-i', required=True,
                       help='Input directory')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory')
    parser.add_argument('--arxiv-id', 
                       help='ArXiv ID')
    parser.add_argument('--batch', action='store_true',
                       help='Process all subdirectories')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--no-dedup', action='store_true',
                       help='Skip deduplication')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger('parser', level=log_level)
    
    # Determine output directory
    if args.output_dir:
        output_base = args.output_dir
    else:
        output_base = args.input_dir
    
    if args.batch:
        # Process all subdirectories
        input_path = Path(args.input_dir)
        all_stats = []
        
        for subdir in input_path.iterdir():
            if not subdir.is_dir():
                continue
            
            # Extract arXiv ID from directory name
            arxiv_id = extract_arxiv_id(str(subdir))
            output_dir = os.path.join(output_base, arxiv_id)
            
            try:
                stats = parse_publication(
                    str(subdir), output_dir, arxiv_id, logger, args
                )
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Error processing {arxiv_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save summary
        summary_path = os.path.join(output_base, 'parsing_summary.json')
        save_json(summary_path, {
            'total_publications': len(all_stats),
            'publications': all_stats
        })
        
        logger.info(f"\n✅ Batch complete: {len(all_stats)} publications")
    
    else:
        # Single publication
        arxiv_id = args.arxiv_id or extract_arxiv_id(args.input_dir)
        output_dir = os.path.join(output_base, arxiv_id) if output_base != args.input_dir else output_base
        
        stats = parse_publication(
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

def extract_arxiv_id(input_dir: str) -> str:
    """Extract arXiv ID from directory path"""
    dir_name = Path(input_dir).name
    
    import re
    pattern = r'\d{4}\.\d{4,5}'
    match = re.search(pattern, dir_name)
    
    if match:
        return match.group()
    
    return dir_name

if __name__ == '__main__':
    main()