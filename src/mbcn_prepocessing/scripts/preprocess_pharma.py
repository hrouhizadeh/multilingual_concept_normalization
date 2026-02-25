#!/usr/bin/env python3
"""
Pharma Dataset Preprocessing

Processes the Spanish Pharmaceutical corpus.

Input Format: BRAT annotation files (.ann)
Output: CSV files with term-CUI mappings

Language: Spanish
Target Ontology: SNOMED CT
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
    flatten_list,
    UMLSDataLoader,
)


class PharmaProcessor:
    """Processor for Spanish Pharma dataset."""
    
    DATASET_NAME = "pharma"
    LANGUAGE = "spa"
    TARGET_ONTOLOGIES = ["snomed"]
    ONTOLOGY_PREFIXES = ["SCT", "SNO"]
    
    def __init__(self, umls_loader: UMLSDataLoader):
        self.umls = umls_loader
        self.snomed_mapping = umls_loader.get_mapping('snomed')
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
                num_dict = {}  # #-annotations: SNOMED codes
                
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        continue
                    
                    if parts[0].startswith("T"):
                        t_dict[parts[0]] = parts[2]
                    elif parts[0].startswith('#'):
                        number = parts[1].split()[1]
                        num_dict[number] = parts[2]
                
                # Match terms with their SNOMED codes
                for t_key, term in t_dict.items():
                    if t_key not in num_dict:
                        continue
                    
                    snomed_code = num_dict[t_key]
                    
                    if snomed_code not in self.snomed_mapping:
                        continue
                    
                    cui_codes = flatten_list([self.snomed_mapping[snomed_code]])
                    
                    # Calculate exact match status
                    em_stat = calculate_exact_match_status(
                        term,
                        self.umls.word_ontology_map,
                        self.train_dev_terms,
                        self.ONTOLOGY_PREFIXES
                    )
                    
                    if data_partition == 'train_dev' and em_stat == 0:
                        em_stat = 2
                    
                    # Extract semantic types and groups
                    sem_types, sem_groups = extract_semantic_types_groups(
                        cui_codes, self.umls.semantic_mapping
                    )
                    
                    pairs.append([term, cui_codes, sem_types, sem_groups, em_stat])
        
        return pairs
    
    def update_train_dev_terms(self, data: list):
        """Add terms to train_dev_terms set."""
        self.train_dev_terms.update(item[0].lower() for item in data)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Pharma dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.PHARMA_DIR,
        help=f"Path to Pharma dataset directory (default: {config.PHARMA_DIR})"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = args.dataset_dir
    source_path = base_path / "source_files"
    
    train_path = source_path / "train"
    dev_path = source_path / "valid"  # Named 'valid' in original
    test_path = source_path / "test"
    
    # Validate directories
    for path in [train_path, dev_path, test_path]:
        if not path.exists():
            print(f"ERROR: Directory not found: {path}")
            sys.exit(1)
    
    # Setup output directories
    config.ensure_directories()
    save_path = config.ensure_dataset_output_dir("Pharma")
    
    # Initialize processor
    print("Loading UMLS data...")
    umls_loader = UMLSDataLoader(config)
    processor = PharmaProcessor(umls_loader)
    
    # Find annotation files
    train_files = find_ann_files(train_path, recursive=False)
    dev_files = find_ann_files(dev_path, recursive=False)
    test_files = find_ann_files(test_path, recursive=False)
    
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
        save_path / "pharma_spa_train.csv",
        train_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        save_path / "pharma_spa_dev.csv",
        dev_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        save_path / "pharma_spa_test.csv",
        test_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    
    # Save to combined directory
    create_csv_file(
        config.TRAIN_OUTPUT_DIR / "pharma_spa_train.csv",
        train_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        config.DEV_OUTPUT_DIR / "pharma_spa_dev.csv",
        dev_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        config.TEST_OUTPUT_DIR / "pharma_spa_test.csv",
        test_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
