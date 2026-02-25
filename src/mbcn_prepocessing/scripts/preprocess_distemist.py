#!/usr/bin/env python3
"""
DisTEMIST Dataset Preprocessing

Processes the DisTEMIST (Disease Text Mining Shared Task) dataset.
Source: https://temu.bsc.es/distemist/

Input Format: TSV files with SNOMED annotations
Output: CSV files with term-CUI mappings

Language: Spanish
Target Ontology: SNOMED CT
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config
from utils import (
    read_json_file,
    create_csv_file,
    extract_semantic_types_groups,
    calculate_exact_match_status,
    split_train_dev,
    find_tsv_files,
    flatten_list,
    UMLSDataLoader,
)


class DisTEMISTProcessor:
    """Processor for DisTEMIST dataset."""
    
    DATASET_NAME = "distemist"
    LANGUAGE = "spa"
    TARGET_ONTOLOGIES = ["snomed"]
    ONTOLOGY_PREFIXES = ["SCT", "SNO"]
    
    def __init__(self, umls_loader: UMLSDataLoader):
        self.umls = umls_loader
        self.snomed_mapping = umls_loader.get_mapping('snomed')
        self.train_dev_terms = set()
    
    def process_tsv_files(self, source_files: list, data_partition: str) -> list:
        """
        Process TSV files with SNOMED annotations.
        
        Args:
            source_files: List of TSV file paths
            data_partition: 'train_dev' or 'test'
            
        Returns:
            List of [term, cui_codes, sem_types, sem_groups, em_stat]
        """
        expanded_rows = []
        
        for sf in source_files:
            print(f"  Processing: {Path(sf).name}")
            df = pd.read_csv(sf, sep='\t')
            
            for _, row in df.iterrows():
                # Only process EXACT matches
                if row.get('semantic_rel') != 'EXACT':
                    continue
                
                term = row['span']
                snomed_code = str(row['code'])
                
                # Map SNOMED to UMLS CUI
                if snomed_code not in self.snomed_mapping:
                    continue
                
                cui_codes = flatten_list([self.snomed_mapping[snomed_code]])
                
                # Placeholder for em_stat (calculated after)
                expanded_rows.append([term, cui_codes, None, None, None])
        
        # Add semantic info and em_stat
        for item in expanded_rows:
            term = item[0]
            cui_codes = item[1]
            
            em_stat = calculate_exact_match_status(
                term,
                self.umls.word_ontology_map,
                self.train_dev_terms,
                self.ONTOLOGY_PREFIXES
            )
            
            if data_partition == 'train_dev' and em_stat == 0:
                em_stat = 2
            
            sem_types, sem_groups = extract_semantic_types_groups(
                cui_codes, self.umls.semantic_mapping
            )
            
            item[2], item[3], item[4] = sem_types, sem_groups, em_stat
        
        return expanded_rows
    
    def update_train_dev_terms(self, data: list):
        """Add terms to train_dev_terms set."""
        self.train_dev_terms.update(item[0].lower() for item in data)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DisTEMIST dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.DISTEMIST_DIR,
        help=f"Path to DisTEMIST dataset directory (default: {config.DISTEMIST_DIR})"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = args.dataset_dir
    source_path = base_path / "source_files"
    
    train_path = source_path / "train"
    test_path = source_path / "test"
    
    # Validate directories
    for path in [train_path, test_path]:
        if not path.exists():
            print(f"ERROR: Directory not found: {path}")
            sys.exit(1)
    
    # Setup output directories
    config.ensure_directories()
    save_path = config.ensure_dataset_output_dir("DisTEMIST")
    
    # Initialize processor
    print("Loading UMLS data...")
    umls_loader = UMLSDataLoader(config)
    processor = DisTEMISTProcessor(umls_loader)
    
    # Find TSV files
    train_files = find_tsv_files(train_path)
    test_files = find_tsv_files(test_path)
    
    print(f"Found files: train={len(train_files)}, test={len(test_files)}")
    
    # Process training data and split into train/dev
    print("Processing training data...")
    train_data = processor.process_tsv_files(train_files, 'train_dev')
    train_set, dev_set = split_train_dev(train_data)
    
    # Update train_dev terms
    processor.update_train_dev_terms(train_set + dev_set)
    
    # Process test data
    print("Processing test data...")
    test_set = processor.process_tsv_files(test_files, 'test')
    
    print(f"\nDataset sizes: train={len(train_set)}, dev={len(dev_set)}, test={len(test_set)}")
    
    # Save to dataset-specific directory
    create_csv_file(
        save_path / "distemist_spa_train.csv",
        train_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        save_path / "distemist_spa_dev.csv",
        dev_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        save_path / "distemist_spa_test.csv",
        test_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    
    # Save to combined directory
    create_csv_file(
        config.TRAIN_OUTPUT_DIR / "distemist_spa_train.csv",
        train_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        config.DEV_OUTPUT_DIR / "distemist_spa_dev.csv",
        dev_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        config.TEST_OUTPUT_DIR / "distemist_spa_test.csv",
        test_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
