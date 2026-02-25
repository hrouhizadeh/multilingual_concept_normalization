#!/usr/bin/env python3
"""
MedUCD Dataset Preprocessing

Processes the French Medication UCD dataset.

Input Format: Excel files (codes.xlsx, labels.xlsx)
Output: CSV files with term-CUI mappings

Language: French
Target Ontology: ATC
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
    split_train_dev_test,
    UMLSDataLoader,
)


class MedUCDProcessor:
    """Processor for French MedUCD dataset."""
    
    DATASET_NAME = "med-ucd"
    LANGUAGE = "fre"
    TARGET_ONTOLOGIES = ["atc"]
    ONTOLOGY_PREFIXES = ["ATC"]
    
    def __init__(self, umls_loader: UMLSDataLoader):
        self.umls = umls_loader
        self.atc_mapping = umls_loader.get_mapping('atc')
        self.train_dev_terms = set()
    
    def process_excel_files(self, code_path: Path, label_path: Path) -> list:
        """
        Process Excel files to extract term-code pairs.
        
        Args:
            code_path: Path to codes.xlsx
            label_path: Path to labels.xlsx
            
        Returns:
            List of [term, umls_codes, sem_types, sem_groups, em_stat_placeholder]
        """
        # Read Excel files
        code_df = pd.read_excel(code_path, engine='openpyxl')
        label_df = pd.read_excel(label_path, engine='openpyxl')
        
        code_df['Source'] = code_df['Source'].astype(str)
        label_df['code'] = label_df['code'].astype(str)
        
        # Create mappings
        source_atc_dict = pd.Series(code_df.Target.values, index=code_df.Source).to_dict()
        term_source_dict = pd.Series(label_df.code.values, index=label_df.CodeLabel).to_dict()
        
        pairs = []
        
        for label, source in term_source_dict.items():
            # Clean term
            term = label.replace('   [7-digit codes]', '').replace('   [13-digit codes]', '')
            
            if source not in source_atc_dict:
                continue
            
            atc_code = source_atc_dict[source]
            if atc_code not in self.atc_mapping:
                continue
            
            umls_codes = self.atc_mapping[atc_code]
            
            # Extract semantic types and groups
            sem_types, sem_groups = extract_semantic_types_groups(
                umls_codes, self.umls.semantic_mapping
            )
            
            # em_stat will be calculated later after split
            pairs.append([term, umls_codes, sem_types, sem_groups, None])
        
        return pairs
    
    def add_em_status(self, data: list, data_partition: str) -> list:
        """Add exact match status to data."""
        for item in data:
            term = item[0]
            em_stat = calculate_exact_match_status(
                term,
                self.umls.word_ontology_map,
                self.train_dev_terms,
                self.ONTOLOGY_PREFIXES
            )
            
            if data_partition == 'train_dev' and em_stat == 0:
                em_stat = 2
            
            item[4] = em_stat
        
        return data
    
    def update_train_dev_terms(self, data: list):
        """Add terms to train_dev_terms set."""
        self.train_dev_terms.update(item[0].lower() for item in data)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MedUCD dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.MED_UCD_DIR,
        help=f"Path to MedUCD dataset directory (default: {config.MED_UCD_DIR})"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = args.dataset_dir
    source_path = base_path / "source_files"
    
    code_path = source_path / "codes.xlsx"
    label_path = source_path / "labels.xlsx"
    
    # Validate input files
    for path in [code_path, label_path]:
        if not path.exists():
            print(f"ERROR: Input file not found: {path}")
            sys.exit(1)
    
    # Setup output directories
    config.ensure_directories()
    save_path = config.ensure_dataset_output_dir("MedUCD")
    
    # Initialize processor
    print("Loading UMLS data...")
    umls_loader = UMLSDataLoader(config)
    processor = MedUCDProcessor(umls_loader)
    
    # Process Excel files
    print("Processing Excel files...")
    all_data = processor.process_excel_files(code_path, label_path)
    
    # Split into train/dev/test (60/20/20)
    train_set, dev_set, test_set = split_train_dev_test(all_data)
    
    # Add exact match status
    train_set = processor.add_em_status(train_set, 'train_dev')
    dev_set = processor.add_em_status(dev_set, 'train_dev')
    
    # Update train_dev terms
    processor.update_train_dev_terms(train_set + dev_set)
    
    test_set = processor.add_em_status(test_set, 'test')
    
    print(f"\nDataset sizes: train={len(train_set)}, dev={len(dev_set)}, test={len(test_set)}")
    
    # Save to dataset-specific directory
    create_csv_file(
        save_path / "ucd_fre_train.csv",
        train_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        save_path / "ucd_fre_dev.csv",
        dev_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        save_path / "ucd_fre_test.csv",
        test_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    
    # Save to combined directory
    create_csv_file(
        config.TRAIN_OUTPUT_DIR / "ucd_fre_train.csv",
        train_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        config.DEV_OUTPUT_DIR / "ucd_fre_dev.csv",
        dev_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        config.TEST_OUTPUT_DIR / "ucd_fre_test.csv",
        test_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
