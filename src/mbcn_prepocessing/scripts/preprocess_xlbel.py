#!/usr/bin/env python3
"""
XL-BEL Dataset Preprocessing

Processes the XL-BEL (Cross-lingual Biomedical Entity Linking) dataset.

Input Format: Text files with CUI||term format
Output: CSV files with term-CUI mappings

Languages: English, German, Spanish, Turkish
Target Ontology: UMLS (direct CUI)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import (
    read_json_file,
    extract_semantic_types_groups,
    calculate_exact_match_simple,
    UMLSDataLoader,
)


class XLBELProcessor:
    """Processor for XL-BEL dataset."""
    
    DATASET_NAME = "xl-bel"
    TARGET_ONTOLOGIES = ["umls"]
    
    LANG_MAP = {
        'en_1k': 'eng',
        'de_1k': 'ger',
        'tr_1k': 'tur',
        'es_1k': 'spa',
    }
    
    def __init__(self, umls_loader: UMLSDataLoader):
        self.umls = umls_loader
        self.cui_codes_set = set(umls_loader.cui_codes)
    
    def process_files(self, file_paths: list) -> dict:
        """
        Process XL-BEL text files.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dict mapping language keys to lists of (cui, term) tuples
        """
        language_data = {key: [] for key in self.LANG_MAP.keys()}
        
        for fp in file_paths:
            if not os.path.exists(fp):
                print(f"Warning: File not found: {fp}")
                continue
            
            with open(fp, 'r', encoding='utf-8') as f:
                concepts = []
                for line in f:
                    parts = line.strip().split('||')
                    if len(parts) >= 2:
                        concepts.append((parts[0].strip(), parts[1].strip()))
            
            # Determine language from filename
            for lang_key in language_data.keys():
                if lang_key in str(fp):
                    language_data[lang_key].extend(concepts)
                    break
        
        return language_data
    
    def write_combined_csv(self, output_path: Path, language_data: dict) -> int:
        """
        Write combined CSV file with all languages.
        
        Args:
            output_path: Output file path
            language_data: Dict from process_files()
            
        Returns:
            Number of rows written
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        seen_terms = set()
        rows_written = 0
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'term', 'code', 'langauge', 'semantic_type', 'semantic_group',
                'targe_ontologies', 'exact_match', 'source'
            ])
            
            for lang_key, terms in language_data.items():
                lang_name = self.LANG_MAP[lang_key]
                print(f"  Processing {lang_name}: {len(terms)} items")
                
                for umls_code, term in terms:
                    # Deduplicate by term+language
                    term_lang = term.lower() + '_' + lang_name
                    if term_lang in seen_terms:
                        continue
                    
                    seen_terms.add(term_lang)
                    
                    # Handle multiple CUIs separated by |
                    if '|' in umls_code:
                        cui_list = umls_code.split('|')
                    else:
                        cui_list = [umls_code]
                    
                    # Filter to valid CUIs
                    valid_cuis = [c for c in cui_list if c in self.cui_codes_set]
                    if not valid_cuis:
                        continue
                    
                    # Calculate exact match status
                    em_stat = calculate_exact_match_simple(term, self.umls.word_ontology_map)
                    
                    # Extract semantic types and groups
                    sem_types, sem_groups = extract_semantic_types_groups(
                        valid_cuis, self.umls.semantic_mapping
                    )
                    
                    writer.writerow([
                        term, valid_cuis, lang_name, sem_types, sem_groups,
                        self.TARGET_ONTOLOGIES, em_stat, self.DATASET_NAME
                    ])
                    rows_written += 1
        
        print(f"Successfully created CSV file: {output_path} ({rows_written} rows)")
        return rows_written


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess XL-BEL dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.XL_BEL_DIR,
        help=f"Path to XL-BEL dataset directory (default: {config.XL_BEL_DIR})"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = args.dataset_dir
    source_path = base_path / "source_files" / "xlbel_v1.0"
    
    if not source_path.exists():
        # Try alternative path
        source_path = base_path / "source_files"
    
    if not source_path.exists():
        print(f"ERROR: Source directory not found: {source_path}")
        sys.exit(1)
    
    # Build file paths
    languages = ['en', 'de', 'tr', 'es']
    file_suffix = '_1k_test_query_with_context.txt'
    files = [source_path / f"{lang}{file_suffix}" for lang in languages]
    
    # Check at least some files exist
    existing_files = [f for f in files if f.exists()]
    if not existing_files:
        print(f"ERROR: No input files found in: {source_path}")
        print(f"Expected files like: en{file_suffix}")
        sys.exit(1)
    
    # Setup output directories
    config.ensure_directories()
    save_path = config.ensure_dataset_output_dir("XL-BEL")
    
    # Initialize processor
    print("Loading UMLS data...")
    umls_loader = UMLSDataLoader(config)
    processor = XLBELProcessor(umls_loader)
    
    # Process files
    print("Processing XL-BEL files...")
    language_data = processor.process_files([str(f) for f in files])
    
    # XL-BEL is test-only dataset
    output_file = save_path / "xl-bel_multi_test.csv"
    processor.write_combined_csv(output_file, language_data)
    
    # Save to combined test directory
    processor.write_combined_csv(
        config.TEST_OUTPUT_DIR / "xl-bel_multi_test.csv",
        language_data
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
