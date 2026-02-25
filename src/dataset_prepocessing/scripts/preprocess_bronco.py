#!/usr/bin/env python3
"""
BRONCO Dataset Preprocessing

Processes the BRONCO150 German clinical corpus.

Input Format: BRAT annotation files (.ann)
Output: CSV files with term-CUI mappings

Language: German
Target Ontologies: ATC, ICD-10
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import (
    read_json_file,
    create_csv_file_dict,
    extract_semantic_types_groups,
    calculate_exact_match_status,
    UMLSDataLoader,
)


class BRONCOProcessor:
    """Processor for BRONCO150 dataset."""
    
    DATASET_NAME = "bronco"
    LANGUAGE = "ger"
    TARGET_ONTOLOGIES = ["atc", "icd-10"]
    ONTOLOGY_PREFIXES = ["ATC", "ICD10"]
    
    def __init__(self, umls_loader: UMLSDataLoader):
        self.umls = umls_loader
        self.icd10_mapping = umls_loader.get_mapping('icd10')
        self.atc_mapping = umls_loader.get_mapping('atc')
        self.train_dev_terms = set()
    
    def get_ann_files(self, folder: Path) -> list:
        """Get all .ann files from folder."""
        return [folder / f for f in os.listdir(folder) if f.endswith('.ann')]
    
    def process_line(self, line: str, concept_dict: dict, mapping_dict: dict):
        """Process a single line from .ann file."""
        parts = line.split('\t')
        if len(parts) < 3:
            return
        
        prefix_num, mapping_code = parts[0], parts[1]
        numeric_id = prefix_num[1:]
        
        if line.startswith('T'):
            concept_dict[numeric_id] = parts[2]
        elif line.startswith('N') and numeric_id not in mapping_dict:
            mapping_code = mapping_code.split()[2]
            mapping_dict[numeric_id] = mapping_code
    
    def pair_lines_from_files(self, file_list: list) -> dict:
        """Extract concept-code pairs from annotation files."""
        paired_lines = {}
        
        for file_path in file_list:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
            
            concept_dict = {}
            mapping_dict = {}
            
            for line in lines:
                if line.startswith(('T', 'N')):
                    self.process_line(line, concept_dict, mapping_dict)
            
            for idx, concept in concept_dict.items():
                mapping_code = mapping_dict.get(idx)
                # Skip OPS codes
                if mapping_code and not mapping_code.startswith('OPS'):
                    paired_lines[str(file_path) + '__' + idx] = (concept, mapping_code)
        
        return paired_lines
    
    def map_to_umls(self, paired_lines: dict, data_partition: str) -> dict:
        """Map source codes to UMLS CUIs."""
        final_dict = {}
        
        for row_id, (concept, source_code) in paired_lines.items():
            source_onto, code = source_code.split(':', 1)
            source_codes = code.split(',') if ',' in code else [code]
            
            # Select appropriate mapping
            if 'ICD' in source_onto:
                relevant_dict = self.icd10_mapping
            else:
                relevant_dict = self.atc_mapping
            
            # Map to UMLS CUIs
            umls_codes = []
            for sc in source_codes:
                umls_codes.extend(relevant_dict.get(sc, []))
            
            if not umls_codes:
                continue
            
            # Calculate exact match status
            em_stat = calculate_exact_match_status(
                concept,
                self.umls.word_ontology_map,
                self.train_dev_terms,
                self.ONTOLOGY_PREFIXES
            )
            
            if data_partition == 'train_dev' and em_stat == 0:
                em_stat = 2
            
            # Extract semantic types and groups
            sem_types, sem_groups = extract_semantic_types_groups(
                umls_codes, self.umls.semantic_mapping
            )
            
            final_dict[row_id] = (concept, umls_codes, sem_types, sem_groups, em_stat)
        
        return final_dict
    
    def update_train_dev_terms(self, instances: dict):
        """Add terms to train_dev_terms set."""
        self.train_dev_terms.update(data[0].lower() for data in instances.values())


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BRONCO dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.BRONCO_DIR,
        help=f"Path to BRONCO dataset directory (default: {config.BRONCO_DIR})"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = args.dataset_dir
    source_folder = base_path / "source_files" / "bratFiles"
    
    if not source_folder.exists():
        print(f"ERROR: Directory not found: {source_folder}")
        sys.exit(1)
    
    # Setup output directories
    config.ensure_directories()
    save_path = config.ensure_dataset_output_dir("BRONCO150")
    
    # Initialize processor
    print("Loading UMLS data...")
    umls_loader = UMLSDataLoader(config)
    processor = BRONCOProcessor(umls_loader)
    
    # Get annotation files
    source_files = sorted(processor.get_ann_files(source_folder))
    
    print(f"Found {len(source_files)} annotation files")
    
    # Split files into train/dev/test (using file indices)
    # BRONCO has limited files, so we split by file
    split_points = [3, 4]
    file_groups = [
        source_files[:split_points[0]],      # train
        [source_files[split_points[0]]],      # dev
        [source_files[split_points[1]]]       # test (remaining)
    ]
    
    if len(source_files) > 5:
        file_groups[2] = source_files[split_points[1]:]
    
    group_names = ['train', 'dev', 'test']
    
    results = {}
    
    for files, group_name in zip(file_groups, group_names):
        print(f"\nProcessing {group_name} set ({len(files)} files)... {files}")
        
        paired_lines = processor.pair_lines_from_files(files)
        
        data_partition = 'train_dev' if group_name in ['train', 'dev'] else 'test'
        instances = processor.map_to_umls(paired_lines, data_partition)
        
        if group_name in ['train', 'dev']:
            processor.update_train_dev_terms(instances)
        
        results[group_name] = instances
        
        # Save to dataset-specific directory
        output_file = save_path / f"bronco_ger_{group_name}.csv"
        rows = create_csv_file_dict(
            output_file, instances,
            processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
        )
        
        # Save to combined directory
        combined_dir = {
            'train': config.TRAIN_OUTPUT_DIR,
            'dev': config.DEV_OUTPUT_DIR,
            'test': config.TEST_OUTPUT_DIR
        }[group_name]
        
        create_csv_file_dict(
            combined_dir / f"bronco_ger_{group_name}.csv",
            instances,
            processor.LANGUAGE, processor.DATASET_NAME, processor.TARGET_ONTOLOGIES
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
