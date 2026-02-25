#!/usr/bin/env python3
"""
Turkish LOINC Dataset Preprocessing

Processes the Turkish Lab Test LOINC Mapping dataset.

Input Format: Excel file with LOINC mappings
Output: CSV files with term-CUI mappings

Language: Turkish
Target Ontology: LOINC
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import config
from utils import (
    read_json_file,
    extract_semantic_types_groups,
    calculate_exact_match_status,
    UMLSDataLoader,
)


class TLLVProcessor:
    """Processor for Turkish LOINC dataset."""
    
    DATASET_NAME = "tllv"
    LANGUAGE = "tur"
    TARGET_ONTOLOGIES = ["lnc"]
    ONTOLOGY_PREFIXES = ['LNC']
    
    def __init__(self, umls_loader: UMLSDataLoader):
        self.umls = umls_loader
        self.loinc_mapping = umls_loader.get_mapping('loinc')
        self.train_dev_terms = set()
    
    def read_source_file(self, file_path: Path) -> pd.DataFrame:
        """Read and process the Turkish LOINC Excel file."""
        # Read Excel file - try both .xls and .xlsx
        try:
            df = pd.read_excel(file_path, sheet_name='SUT all', index_col=None)
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            raise
        
        df = df.drop_duplicates()
        
        # Get column names (handle encoding issues)
        cols = df.columns.tolist()
        
        # Build term from multiple columns (columns 1-8 typically)
        term_cols = df.iloc[:, 1:9].fillna('').astype(str)
        result = term_cols.agg(' '.join, axis=1)
        
        # Find LOINC ID column
        loinc_col = None
        for col in cols:
            if 'LOINC' in str(col).upper() and ('ID' in str(col).upper() or 'NUMAR' in str(col).upper()):
                loinc_col = col
                break
        
        if loinc_col is None:
            # Fallback to last column
            loinc_col = cols[-1]
        
        output_df = pd.DataFrame({
            'term': result,
            'code': df[loinc_col]
        })
        
        output_df = output_df.dropna()
        
        # Deduplicate by lowercase term
        output_df['term_lower'] = output_df['term'].astype(str).str.lower()
        output_df = output_df.drop_duplicates(subset=['term_lower'], keep='first')
        output_df = output_df.drop(columns=['term_lower'])
        
        # Map LOINC codes to UMLS CUIs
        output_df['code'] = output_df['code'].astype(str).map(self.loinc_mapping)
        
        # Remove rows where mapping failed
        output_df = output_df.dropna(subset=['code'])
        
        # Shuffle
        output_df = output_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return output_df
    

    
    def create_csv_from_df(self, output_path: Path, df: pd.DataFrame) -> int:
        """Create CSV file from DataFrame."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        seen_terms = set()
        rows_written = 0
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'term', 'code', 'langauge', 'semantic_type', 'semantic_group',
                'targe_ontologies', 'exact_match', 'source'
            ])
            for _, row in df.iterrows():
                term = row['term']
                code = row['code']

                term_lower = str(term).lower()
                if term_lower in seen_terms:
                    continue
                
                seen_terms.add(term_lower)

                em_stat = calculate_exact_match_status(
                        term,
                        self.umls.word_ontology_map,
                        self.train_dev_terms,
                        self.ONTOLOGY_PREFIXES
                    )
                if ('tllv_tur_dev' in output_path.name or  'tllv_tur_train' in output_path.name) and em_stat == 0:
                    em_stat = 2                
                    
                # Handle code as list if needed
                code_list = code if isinstance(code, list) else [code]
                sem_types, sem_groups = extract_semantic_types_groups(
                    code_list, self.umls.semantic_mapping
                )
                
                writer.writerow([
                    term, code, self.LANGUAGE, sem_types, sem_groups,
                    self.TARGET_ONTOLOGIES, em_stat, self.DATASET_NAME
                ])
                rows_written += 1
        
        print(f"Successfully created CSV file: {output_path} ({rows_written} rows)")
        return rows_written


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Turkish LOINC dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.TLLV_DIR,
        help=f"Path to Turkish dataset directory (default: {config.TLLV_DIR})"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = args.dataset_dir
    source_path = base_path / "source_files"
    
    # Look for Excel file
    source_file = None
    for ext in ['.xls', '.xlsx']:
        for f in source_path.glob(f'*{ext}'):
            source_file = f
            break
    
    if source_file is None or not source_file.exists():
        print(f"ERROR: No Excel file found in: {source_path}")
        sys.exit(1)
    
    # Setup output directories
    config.ensure_directories()
    save_path = config.ensure_dataset_output_dir("TLLV")
    
    # Initialize processor
    print("Loading UMLS data...")
    umls_loader = UMLSDataLoader(config)
    processor = TLLVProcessor(umls_loader)
    
    # Read and process source file
    print(f"Processing: {source_file}")
    source_df = processor.read_source_file(source_file)
    
    print(f"Total records after processing: {len(source_df)}")
        
    train_df, dev_df, test_df = np.split(source_df.sample(frac=1, random_state=42), [int(.6*len(source_df)), int(.8*len(source_df))])

    print(f"\nDataset sizes: train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}")
    
    # Save to dataset-specific directory
    processor.create_csv_from_df(save_path / "tllv_tur_train.csv", train_df)
    processor.create_csv_from_df(save_path / "tllv_tur_dev.csv", dev_df)
    processor.create_csv_from_df(save_path / "tllv_tur_test.csv", test_df)
    
    # Save to combined directory
    processor.create_csv_from_df(config.TRAIN_OUTPUT_DIR / "tllv_tur_train.csv", train_df)
    processor.create_csv_from_df(config.DEV_OUTPUT_DIR / "tllv_tur_dev.csv", dev_df)
    processor.create_csv_from_df(config.TEST_OUTPUT_DIR / "tllv_tur_test.csv", test_df)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
