"""
Main script to run parsing pipeline
"""

import argparse
from parser import *
from utils import *
import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', default=config.OUTPUT_DIR)
    args = parser.parse_args()
    
    # Step 1: Gather files
    print("Step 1: Gathering files...")
    gatherer = FileGatherer(args.input_dir)
    version_files = gatherer.get_all_version_files()
    
    # Step 2: Build hierarchy
    print("Step 2: Building hierarchy...")
    builder = HierarchyBuilder(arxiv_id)
    for version, files in version_files.items():
        latex_content = combine_files(files)
        builder.parse_latex(latex_content, version)
    
    # Step 3: Clean and normalize
    print("Step 3: Cleaning...")
    cleaner = LaTeXCleaner()
    # ... clean elements
    
    # Step 4: Extract references
    print("Step 4: Extracting references...")
    ref_extractor = ReferenceExtractor()
    # ... extract
    
    # Step 5: Deduplicate
    print("Step 5: Deduplicating...")
    deduper = Deduplicator()
    # ... deduplicate
    
    # Step 6: Save outputs
    print("Step 6: Saving outputs...")
    save_json(f'{output_dir}/hierarchy.json', builder.get_hierarchy_json())
    ref_extractor.save_to_bib_file(f'{output_dir}/refs.bib')
    
    print("✅ Parsing complete!")

if __name__ == '__main__':
    main()