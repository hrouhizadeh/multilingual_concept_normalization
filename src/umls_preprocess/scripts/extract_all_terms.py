#!/usr/bin/env python3
"""
UMLS all terms extractor

Reads the UMLS MRCONSO.RRF file and creates a JSONL (JSON Lines) file
with comprehensive term data for each CUI.

MRCONSO.RRF Format (pipe-delimited):
    Column 0:  CUI  - Concept Unique Identifier
    Column 1:  LAT  - Language
    Column 11: SAB  - Source Abbreviation
    Column 13: CODE - Source Code
    Column 14: STR  - String (the actual term)

Output format (JSON Lines - one JSON object per line):
    {"CUI": "C0000039", "MSH": [...], "SNOMEDCT_US": [...], "terms_ENG": "...", "terms_all": "..."}
    {"CUI": "C0000052", "MSH": [...], "terms_ENG": "...", "terms_all": "..."}

Each line contains:
    - CUI: The concept identifier
    - {SAB}: Array of {CODE, STR, LAT} for each source
    - terms_{LAT}: Space-joined terms for each language
    - terms_all: Space-joined all terms

Usage:
    python extract_all_terms.py
    python extract_all_terms.py --input /path/to/MRCONSO.RRF --output terms.jsonl
"""

import argparse
import sys
from pathlib import Path

try:
    import jsonlines
except ImportError:
    print("ERROR: jsonlines package required. Install with: pip install jsonlines")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas package required. Install with: pip install pandas")
    sys.exit(1)

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MRCONSO_FILE,
    ALL_TERMS_OUTPUT,
    ensure_directories,
    validate_input_file,
)


# MRCONSO.RRF column names (includes DUMMY for trailing pipe)
MRCONSO_COLUMNS = [
    "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI",
    "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL",
    "SUPPRESS", "CVF", "DUMMY"
]


def extract_all_terms(input_file: Path, output_file: Path) -> None:
    """
    Extract all terms from MRCONSO.RRF and save to JSONL format.
    
    Args:
        input_file: Path to MRCONSO.RRF file
        output_file: Path for output JSONL file
    """
    print(f"Loading {input_file}...")
    print("  (This may take a few minutes for large files)")
    
    # Load MRCONSO.RRF with pandas
    df = pd.read_table(
        input_file,
        sep='|',
        names=MRCONSO_COLUMNS,
        index_col=False,
        low_memory=False,
        quoting=3  # QUOTE_NONE - handle fields with quotes
    )
    
    print(f"  Loaded {len(df):,} rows")
    
    # Keep only needed columns
    df = df[["CUI", "SAB", "CODE", "STR", "LAT"]]
    
    print("Grouping by CUI and processing...")
    
    # Process and write to JSONL
    cui_count = 0
    with jsonlines.open(output_file, 'w') as output:
        grouped_by_cui = df.groupby("CUI")
        total_cuis = len(grouped_by_cui)
        
        for cui, group in grouped_by_cui:
            output_record = {"CUI": cui}
            
            # Process Source (SAB) groups
            terms_all_list = []
            for sab, sab_group in group.groupby("SAB"):
                # Convert group to records, excluding SAB and CUI columns
                sab_data = sab_group.drop(['SAB', 'CUI'], axis=1).to_dict(orient='records')
                output_record[sab] = sab_data
                terms_all_list.extend(sab_group['STR'].astype(str).tolist())
            
            # Process Language (LAT) groups
            for lat, lat_group in group.groupby("LAT"):
                terms_lat_list = lat_group['STR'].astype(str).tolist()
                output_record['terms_' + lat] = " ".join(terms_lat_list)
            
            # Add all terms concatenated
            output_record['terms_all'] = " ".join(terms_all_list)
            
            output.write(output_record)
            
            cui_count += 1
            if cui_count % 100000 == 0:
                print(f"  Processed {cui_count:,}/{total_cuis:,} CUIs ({100*cui_count/total_cuis:.1f}%)")
    
    print(f"Done! Processed {cui_count:,} CUIs")
    print(f"Output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract all terms from UMLS MRCONSO.RRF to JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_all_terms.py
  python extract_all_terms.py --input /path/to/MRCONSO.RRF --output all_terms.jsonl

Note: This script requires significant memory for large UMLS releases.
      Recommend at least 16GB RAM for full UMLS processing.
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=MRCONSO_FILE,
        help=f"Path to MRCONSO.RRF file (default: {MRCONSO_FILE})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=ALL_TERMS_OUTPUT,
        help=f"Path for output JSONL file (default: {ALL_TERMS_OUTPUT})"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_directories()
    
    # Validate input file
    if not validate_input_file(args.input, "MRCONSO.RRF"):
        sys.exit(1)
    
    # Run extraction
    extract_all_terms(args.input, args.output)


if __name__ == "__main__":
    main()
