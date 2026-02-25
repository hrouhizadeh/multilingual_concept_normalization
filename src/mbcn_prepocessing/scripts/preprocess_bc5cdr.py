#!/usr/bin/env python3
"""
BC5CDR Dataset Preprocessing

Processes the BioCreative V CDR (Chemical Disease Relation) corpus.
Source: https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/

Input Format: PubTator format files
Output: CSV files with term-CUI mappings

Language: English
Target Ontology: MeSH
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import (
    read_json_file,
    create_csv_file,
    extract_semantic_types_groups,
    calculate_exact_match_status,
    flatten_list,
    UMLSDataLoader,
)


class BC5CDRProcessor:
    """Processor for BC5CDR dataset."""
    
    DATASET_NAME = "bc5cdr"
    LANGUAGE = "eng"
    TARGET_ONTOLOGIES = ["mesh"]
    ONTOLOGY_PREFIXES = ["MSH"]
    
    def __init__(self, umls_loader: UMLSDataLoader):
        self.umls = umls_loader
        self.mesh_mapping = umls_loader.get_mapping('mesh')
        self.train_dev_terms = set()
    
    def process_file(self, file_path: Path, data_partition: str) -> list:
        """
        Process a PubTator format file.
        
        Args:
            file_path: Path to PubTator file
            data_partition: 'train_dev' or 'test'
            
        Returns:
            List of [term, umls_codes, sem_types, sem_groups, em_stat]
        """
        data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip title and abstract lines
                if '|t|' in line or '|a|' in line:
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) != 6:
                    continue
                
                term = parts[3]
                mesh_code = parts[5]
                
                # Map MeSH codes to UMLS CUIs
                umls_codes = self._map_mesh_to_umls(mesh_code)
                if not umls_codes:
                    continue
                
                # Calculate exact match status
                em_stat = calculate_exact_match_status(
                    term,
                    self.umls.word_ontology_map,
                    self.train_dev_terms,
                    self.ONTOLOGY_PREFIXES
                )
                
                # For train/dev, mark unseen terms
                if data_partition == 'train_dev' and em_stat == 0:
                    em_stat = 2
                
                # Extract semantic types and groups
                sem_types, sem_groups = extract_semantic_types_groups(
                    umls_codes, self.umls.semantic_mapping
                )
                
                data.append([term, umls_codes, sem_types, sem_groups, em_stat])
        
        return data
    
    def _map_mesh_to_umls(self, mesh_code: str) -> list:
        """Map MeSH code(s) to UMLS CUI(s)."""
        umls_codes = []
        
        if '|' in mesh_code:
            # Multiple MeSH codes
            for code in mesh_code.split('|'):
                if code in self.mesh_mapping:
                    umls_codes.extend(self.mesh_mapping[code])
        elif mesh_code in self.mesh_mapping:
            umls_codes = self.mesh_mapping[mesh_code]
        
        return umls_codes
    
    def update_train_dev_terms(self, data: list):
        """Add terms to train_dev_terms set."""
        self.train_dev_terms.update(item[0].lower() for item in data)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BC5CDR dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.BC5CDR_DIR,
        help=f"Path to BC5CDR dataset directory (default: {config.BC5CDR_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = args.dataset_dir
    source_path = base_path / "source_files"
    
    train_path = source_path / "CDR_TrainingSet.PubTator.txt"
    dev_path = source_path / "CDR_DevelopmentSet.PubTator.txt"
    test_path = source_path / "CDR_TestSet.PubTator.txt"
    
    # Validate input files
    for path in [train_path, dev_path, test_path]:
        if not path.exists():
            print(f"ERROR: Input file not found: {path}")
            print(f"\nPlease ensure BC5CDR source files are in: {source_path}")
            sys.exit(1)
    
    # Setup output directories
    config.ensure_directories()
    save_path = args.output_dir or config.ensure_dataset_output_dir("BC5CDR")
    
    # Initialize processor
    print("Loading UMLS data...")
    umls_loader = UMLSDataLoader(config)
    processor = BC5CDRProcessor(umls_loader)
    
    # Process train and dev sets
    print("Processing training set...")
    train_set = processor.process_file(train_path, 'train_dev')
    
    print("Processing development set...")
    dev_set = processor.process_file(dev_path, 'train_dev')
    
    # Update train_dev terms for exact match calculation
    processor.update_train_dev_terms(train_set + dev_set)
    
    # Process test set
    print("Processing test set...")
    test_set = processor.process_file(test_path, 'test')
    
    print(f"\nDataset sizes: train={len(train_set)}, dev={len(dev_set)}, test={len(test_set)}")
    
    # Save to dataset-specific directory
    create_csv_file(
        save_path / "bc5cdr_eng_train.csv",
        train_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        save_path / "bc5cdr_eng_dev.csv",
        dev_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        save_path / "bc5cdr_eng_test.csv",
        test_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    
    # Save to combined directory
    create_csv_file(
        config.TRAIN_OUTPUT_DIR / "bc5cdr_eng_train.csv",
        train_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        config.DEV_OUTPUT_DIR / "bc5cdr_eng_dev.csv",
        dev_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    create_csv_file(
        config.TEST_OUTPUT_DIR / "bc5cdr_eng_test.csv",
        test_set, processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
