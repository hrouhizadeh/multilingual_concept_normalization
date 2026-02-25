#!/usr/bin/env python3
"""
GSC/MANTRA Dataset Preprocessing

Processes the Gold Standard Corpus (GSC) from the MANTRA project.
Multilingual medical entity annotations.

Input Format: XML files with CUI annotations
Output: CSV files with term-CUI mappings

Languages: English, German, Spanish, French
Target Ontologies: MedDRA, MeSH, SNOMED
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import (
    read_json_file,
    create_csv_file_multilang,
    extract_semantic_types_groups,
    calculate_exact_match_simple,
    UMLSDataLoader,
)


class GSCProcessor:
    """Processor for GSC/MANTRA dataset."""
    
    DATASET_NAME = "gsc"
    TARGET_ONTOLOGIES = ["meddra", "mesh", "snomed"]
    ONTOLOGY_PREFIXES = ["SNO", "SCT", "MDR", "MSH"]
    
    LANG_MAP = {
        'de': 'ger',
        'en': 'eng',
        'es': 'spa',
        'fr': 'fre'
    }
    
    def __init__(self, umls_loader: UMLSDataLoader):
        self.umls = umls_loader
        self.cui_codes_set = set(umls_loader.cui_codes)
    
    def parse_xml_file(self, file_path: Path, file_lang: str, source: str) -> list:
        """
        Parse XML file and extract term-CUI data.
        
        Args:
            file_path: Path to XML file
            file_lang: Two-letter language code
            source: Source identifier
            
        Returns:
            List of [term, [cui], lang, sem_types, sem_groups, ontologies, em_stat, source]
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        data = []
        seen_terms = set()
        
        lang_code = self.LANG_MAP.get(file_lang, file_lang)
        
        for element in root.iter('e'):
            cui = element.attrib.get('cui')
            term = element.text
            
            # Validate CUI
            if cui not in self.cui_codes_set:
                continue
            
            if not cui or not term:
                continue
            
            # Deduplicate by term+language
            term_lang = term.lower() + '_' + lang_code
            if term_lang in seen_terms:
                continue
            
            seen_terms.add(term_lang)
            
            # Calculate exact match status
            em_stat = self._calculate_em_status(term)
            
            # Extract semantic types and groups
            sem_types, sem_groups = extract_semantic_types_groups(
                [cui], self.umls.semantic_mapping
            )
            
            data.append([
                term, [cui], lang_code, sem_types, sem_groups,
                self.TARGET_ONTOLOGIES, em_stat, source
            ])
        
        return data
    
    def _calculate_em_status(self, term: str) -> int:
        """Calculate exact match status for GSC."""
        term_lower = term.lower()
        mappings = self.umls.word_ontology_map.get(term_lower, [])
        
        in_umls = any(
            m.startswith(prefix) for m in mappings
            for prefix in self.ONTOLOGY_PREFIXES
        )
        
        return 1 if in_umls else 0
    
    def process_directory(self, base_dir: Path, subdirs: list) -> list:
        """
        Process all XML files in given subdirectories.
        
        Args:
            base_dir: Base directory
            subdirs: List of subdirectory names
            
        Returns:
            Combined list of all extracted data
        """
        all_data = []
        
        for subdir in subdirs:
            full_path = base_dir / subdir
            
            if not full_path.exists():
                print(f"Warning: Directory not found: {full_path}")
                continue
            
            for filename in os.listdir(full_path):
                if not filename.endswith('.xml'):
                    continue
                
                # Extract language from filename (format: SOURCE_TYPE_LANG.xml)
                parts = filename.split('_')
                if len(parts) >= 3:
                    file_lang = parts[2].replace('.xml', '')
                    source = f"gsc-{parts[0].lower()}"
                else:
                    continue
                
                if file_lang not in self.LANG_MAP:
                    continue
                
                print(f"  Processing: {filename} ({file_lang})")
                
                file_path = full_path / filename
                data = self.parse_xml_file(file_path, file_lang, source)
                all_data.extend(data)
        
        return all_data


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess GSC/MANTRA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.GSC_DIR,
        help=f"Path to GSC dataset directory (default: {config.GSC_DIR})"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = args.dataset_dir
    source_path = base_path / "source_files"
    
    subdirs = ['EMEA', 'MEDLINE', 'PATENT']
    
    # Validate at least one directory exists
    valid_dirs = [d for d in subdirs if (source_path / d).exists()]
    if not valid_dirs:
        print(f"ERROR: No source directories found in: {source_path}")
        print(f"Expected subdirectories: {subdirs}")
        sys.exit(1)
    
    # Setup output directories
    config.ensure_directories()
    save_path = config.ensure_dataset_output_dir("GSC")
    
    # Initialize processor
    print("Loading UMLS data...")
    umls_loader = UMLSDataLoader(config)
    processor = GSCProcessor(umls_loader)
    
    # Process all directories
    print("Processing XML files...")
    all_data = processor.process_directory(source_path, subdirs)
    
    # Deduplicate by term+language across all sources
    final_data = []
    seen_terms = set()
    
    for item in all_data:
        term_lang = item[0].lower() + '_' + item[2]
        if term_lang not in seen_terms:
            seen_terms.add(term_lang)
            final_data.append(item)
    
    print(f"\nTotal unique entries: {len(final_data)}")
    
    # GSC is test-only dataset
    output_file = save_path / "gsc_multi_test.csv"
    create_csv_file_multilang(output_file, final_data)
    
    # Save to combined test directory
    create_csv_file_multilang(
        config.TEST_OUTPUT_DIR / "gsc_multi_test.csv",
        final_data
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
