#!/usr/bin/env python3
"""
Quaero Dataset Preprocessing

Processes the QUAERO French Medical Corpus.
Source: https://quaerofrenchmed.limsi.fr/

Input Format: BRAT annotation files (.ann)
Output: CSV files with term-CUI mappings

Language: French
Target Ontology: UMLS (direct CUI)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import (
    read_json_file,
    create_csv_file,
    extract_semantic_types_groups,
    calculate_exact_match_status,
    find_ann_files,
    UMLSDataLoader,
)


class QuaeroProcessor:
    """Processor for Quaero dataset."""
    
    DATASET_NAME = "quaero"
    LANGUAGE = "fre"
    ONTOLOGY_PREFIXES = ['']
    TARGET_ONTOLOGIES = ["umls"]
    
    def __init__(self, umls_loader: UMLSDataLoader):
        self.umls = umls_loader
        self.cui_codes_set = set(umls_loader.cui_codes)
        self.train_dev_terms = set()
    
    def process_ann_files(self, source_files: list, data_partition: str) -> list:
        """
        Process BRAT .ann files.
        
        Args:
            source_files: List of .ann file paths
            data_partition: 'train_dev' or 'test'
            
        Returns:
            List of [term, cui_codes, sem_types, sem_groups, em_stat]
        """
        pairs = []
        
        for sf in source_files:
            with open(sf, 'r', encoding='utf-8') as f:
                t_dict = {}  # T-annotations: term text
                num_dict = {}  # #-annotations: CUI codes
                
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        continue
                    
                    key, middle_part, value = parts
                    
                    if key.startswith("T"):
                        t_dict[key] = value
                    elif key.startswith('#'):
                        number = middle_part.split()[1]
                        num_dict[number] = value
                
                # Match terms with their CUI codes
                for t_key, term in t_dict.items():
                    if t_key not in num_dict:
                        continue
                    
                    codes_string = num_dict[t_key]
                    
                    # Parse CUI codes (space or comma separated)
                    if ' ' in codes_string:
                        cui_codes = codes_string.split(' ')
                    elif ',' in codes_string:
                        cui_codes = codes_string.split(',')
                    else:
                        cui_codes = [codes_string]
                    
                    # Filter to valid CUIs
                    valid_cuis = [c for c in cui_codes if c in self.cui_codes_set]
                    if not valid_cuis:
                        continue
                    
                    em_stat = calculate_exact_match_status(
                            term,
                            self.umls.word_ontology_map,
                            self.train_dev_terms,
                            self.ONTOLOGY_PREFIXES)
                            
                    if data_partition == 'train_dev' and em_stat == 0:
                        em_stat = 2
                    
                    # Extract semantic types and groups
                    sem_types, sem_groups = extract_semantic_types_groups(
                        valid_cuis, self.umls.semantic_mapping
                    )
                    
                    pairs.append([term, valid_cuis, sem_types, sem_groups, em_stat])
        
        return pairs
    
    # def _calculate_em_status(self, term: str, data_partition: str) -> int:
    #     """Calculate exact match status for Quaero."""
    #     term_lower = term.lower()
    #     in_train_dev = term_lower in self.train_dev_terms
    #     in_umls = bool(self.umls.word_ontology_map.get(term_lower))
        
    #     if in_train_dev and in_umls:
    #         return 3
    #     elif in_train_dev:
    #         return 2
    #     elif in_umls:
    #         return 1
        
    #     # For train/dev, mark unseen terms
    #     if data_partition == 'train_dev':
    #         return 2
    #     return 0
    
    def update_train_dev_terms(self, data: list):
        """Add terms to train_dev_terms set."""
        self.train_dev_terms.update(item[0].lower() for item in data)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Quaero dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.QUAERO_DIR,
        help=f"Path to Quaero dataset directory (default: {config.QUAERO_DIR})"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = args.dataset_dir
    source_path = base_path / "source_files"
    
    train_path = source_path / "train"
    dev_path = source_path / "dev"
    test_path = source_path / "test"
    
    # Validate directories
    for path in [train_path, dev_path, test_path]:
        if not path.exists():
            print(f"ERROR: Directory not found: {path}")
            sys.exit(1)
    
    # Setup output directories
    config.ensure_directories()
    save_path = config.ensure_dataset_output_dir("Quaero")
    
    # Initialize processor
    print("Loading UMLS data...")
    umls_loader = UMLSDataLoader(config)
    processor = QuaeroProcessor(umls_loader)
    
    # Find annotation files
    train_files = find_ann_files(train_path)
    dev_files = find_ann_files(dev_path)
    test_files = find_ann_files(test_path)
    
    print(f"Found files: train={len(train_files)}, dev={len(dev_files)}, test={len(test_files)}")
    
    # Process train and dev sets
    print("Processing training set...")
    train_set = processor.process_ann_files(train_files, 'train_dev')
    
    print("Processing development set...")
    dev_set = processor.process_ann_files(dev_files, 'train_dev')
    
    # Update train_dev terms
    processor.update_train_dev_terms(train_set + dev_set)
    
    # Process test set
    print("Processing test set...")
    test_set = processor.process_ann_files(test_files, 'test')
    
    print(f"\nDataset sizes: train={len(train_set)}, dev={len(dev_set)}, test={len(test_set)}")
    
    # Save to dataset-specific directory
    create_csv_file(
        save_path / "quaero_fre_train.csv",
        train_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        save_path / "quaero_fre_dev.csv",
        dev_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        save_path / "quaero_fre_test.csv",
        test_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    
    # Save to combined directory
    create_csv_file(
        config.TRAIN_OUTPUT_DIR / "quaero_fre_train.csv",
        train_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        config.DEV_OUTPUT_DIR / "quaero_fre_dev.csv",
        dev_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        config.TEST_OUTPUT_DIR / "quaero_fre_test.csv",
        test_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
