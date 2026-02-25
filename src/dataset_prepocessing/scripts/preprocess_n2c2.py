#!/usr/bin/env python3
"""
N2C2 Dataset Preprocessing

Processes the n2c2 (National NLP Clinical Challenges) dataset.
Source: https://n2c2.dbmi.hms.harvard.edu/

Input Format: .norm and .txt paired files
Output: CSV files with term-CUI mappings

Language: English
Target Ontologies: RxNorm, SNOMED
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import (
    read_json_file,
    create_csv_file,
    extract_semantic_types_groups,
    calculate_exact_match_status,
    split_train_dev,
    UMLSDataLoader,
)


class N2C2Processor:
    """Processor for N2C2 dataset."""
    
    DATASET_NAME = "n2c2"
    LANGUAGE = "eng"
    TARGET_ONTOLOGIES = ["rxnorm", "snomed"]
    ONTOLOGY_PREFIXES = ["RX", "SNO"]
    
    def __init__(self, umls_loader: UMLSDataLoader):
        self.umls = umls_loader
        self.train_dev_terms = set()

    def pair_matched_files(self, norm_folder: Path, note_folder: Path) -> list:
        norm_files = {f.split('.')[0]: f for f in os.listdir(norm_folder) if f.endswith('.norm')}
        txt_files = {f.split('.')[0]: f for f in os.listdir(note_folder) if f.endswith('.txt')}
        
        # Sort base names for deterministic order
        paired = [
            (norm_folder / norm_files[base], note_folder / txt_files[base])
            for base in sorted(norm_files.keys())  # <-- Add sorted()
            if base in txt_files
        ]
        
        return paired    

    def process_paired_files(self, matched_files: list, data_partition: str) -> list:
        """
        Process paired .norm and .txt files.
        
        Args:
            matched_files: List of (norm_path, txt_path) tuples
            data_partition: 'train_dev' or 'test'
            
        Returns:
            List of [term, cui_codes, sem_types, sem_groups, em_stat]
        """
        content = []
        
        for norm_path, txt_path in matched_files:
            # Read the clinical note
            with open(txt_path, 'r', encoding='utf-8') as f:
                notes = f.read()
            
            # Read the normalization annotations
            with open(norm_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('||')
                    cui_code = parts[1]
                    
                    if cui_code == 'CUI-less':
                        continue
                    
                    # Extract term from character offsets
                    if len(parts) > 4:
                        # Discontinuous mention
                        s1, e1, s2, e2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                        term = notes[s1:e1] + ' ' + notes[s2:e2]
                    else:
                        # Continuous mention
                        start, end = int(parts[2]), int(parts[3])
                        term = notes[start:end]
                    
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
                        [cui_code], self.umls.semantic_mapping
                    )
                    
                    content.append([term, [cui_code], sem_types, sem_groups, em_stat])
        
        return content
    
    def update_train_dev_terms(self, data: list):
        """Add terms to train_dev_terms set."""
        self.train_dev_terms.update(item[0].lower() for item in data)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess N2C2 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.N2C2_DIR,
        help=f"Path to N2C2 dataset directory (default: {config.N2C2_DIR})"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = args.dataset_dir
    source_path = base_path / "source_files"
    
    train_norm_folder = source_path / "train_norm"
    train_note_folder = source_path / "train_note"
    test_norm_folder = source_path / "test_norm"
    test_note_folder = source_path / "test_note"
    
    # Validate directories
    for folder in [train_norm_folder, train_note_folder, test_norm_folder, test_note_folder]:
        if not folder.exists():
            print(f"ERROR: Directory not found: {folder}")
            sys.exit(1)
    
    # Setup output directories
    config.ensure_directories()
    save_path = config.ensure_dataset_output_dir("N2C2")
    
    # Initialize processor
    print("Loading UMLS data...")
    umls_loader = UMLSDataLoader(config)
    processor = N2C2Processor(umls_loader)
    
    # Process training data
    print("Processing training data...")
    train_matched = processor.pair_matched_files(train_norm_folder, train_note_folder)
    train_data = processor.process_paired_files(train_matched, 'train_dev')
    
    # Split train into train/dev
    train_set, dev_set = split_train_dev(train_data)
    
    # Update train_dev terms
    processor.update_train_dev_terms(train_set + dev_set)
    
    # Process test data
    print("Processing test data...")
    test_matched = processor.pair_matched_files(test_norm_folder, test_note_folder)
    test_set = processor.process_paired_files(test_matched, 'test')
    
    print(f"\nDataset sizes: train={len(train_set)}, dev={len(dev_set)}, test={len(test_set)}")
    
    # Save to dataset-specific directory
    create_csv_file(
        save_path / "n2c2_eng_train.csv",
        train_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        save_path / "n2c2_eng_dev.csv",
        dev_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        save_path / "n2c2_eng_test.csv",
        test_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    
    # Save to combined directory
    create_csv_file(
        config.TRAIN_OUTPUT_DIR / "n2c2_eng_train.csv",
        train_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        config.DEV_OUTPUT_DIR / "n2c2_eng_dev.csv",
        dev_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        config.TEST_OUTPUT_DIR / "n2c2_eng_test.csv",
        test_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
