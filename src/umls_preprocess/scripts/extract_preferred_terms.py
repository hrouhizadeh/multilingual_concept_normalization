#!/usr/bin/env python3
"""
UMLS preferred term extractor

Reads the UMLS MRCONSO.RRF file and creates a JSON file with CUI as key
and preferred term as value.

MRCONSO.RRF Format (pipe-delimited):
    Column 0:  CUI     - Concept Unique Identifier
    Column 1:  LAT     - Language
    Column 2:  TS      - Term Status (P = Preferred)
    Column 3:  LUI     - Lexical Unique Identifier
    Column 4:  STT     - String Type (PF = Preferred Form)
    Column 5:  SUI     - String Unique Identifier
    Column 6:  ISPREF  - Is Preferred (Y = Yes)
    Column 7:  AUI     - Atom Unique Identifier
    Column 8:  SAUI    - Source Atom Unique Identifier
    Column 9:  SCUI    - Source Concept Unique Identifier
    Column 10: SDUI    - Source Descriptor Unique Identifier
    Column 11: SAB     - Source Abbreviation
    Column 12: TTY     - Term Type
    Column 13: CODE    - Source Code
    Column 14: STR     - String (the actual term)
    Column 15: SRL     - Source Restriction Level
    Column 16: SUPPRESS - Suppressibility
    Column 17: CVF     - Content View Flag

Preferred term criteria:
    - LAT = 'ENG' (English, or specified language)
    - TS = 'P' (Preferred Term Status)
    - STT = 'PF' (Preferred Form)
    - ISPREF = 'Y' (Is Preferred)

Output format:
    {
        "C0000039": "1,2-dipalmitoylphosphatidylcholine",
        "C0000052": "1,4-alpha-Glucan Branching Enzyme",
        ...
    }

Usage:
    python extract_preferred_terms.py
    python extract_preferred_terms.py --language SPA  # For Spanish
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MRCONSO_FILE,
    PREFERRED_TERMS_OUTPUT,
    ensure_directories,
    validate_input_file,
)


def extract_preferred_terms(input_file: Path, output_file: Path, language: str = 'ENG') -> dict:
    """
    Extract preferred terms from MRCONSO.RRF file.
    
    Args:
        input_file: Path to MRCONSO.RRF file
        output_file: Path for output JSON file
        language: Language code to filter by (default: 'ENG')
    
    Returns:
        Dictionary with CUI as key and preferred term as value
    """
    cui_terms = {}
    
    print(f"Reading {input_file}...")
    print(f"Filtering for language: {language}")
    
    line_count = 0
    matched_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            
            line_count += 1
            
            # Split by pipe delimiter
            parts = line.split('|')
            
            if len(parts) < 15:
                continue
            
            cui = parts[0]       # Concept Unique Identifier
            lat = parts[1]       # Language
            ts = parts[2]        # Term Status
            stt = parts[4]       # String Type
            ispref = parts[6]    # Is Preferred
            term = parts[14]     # String (the actual term)
            
            # Check for preferred term criteria
            if lat == language and ts == 'P' and stt == 'PF' and ispref == 'Y':
                # Only keep the first preferred term found for each CUI
                if cui not in cui_terms:
                    cui_terms[cui] = term
                    matched_count += 1
            
            # Progress indicator for large files
            if line_count % 500000 == 0:
                print(f"  Processed {line_count:,} lines, found {matched_count:,} preferred terms...")
    
    print(f"Finished processing {line_count:,} lines")
    print(f"Found {len(cui_terms):,} unique CUIs with preferred terms")
    
    # Write JSON output
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cui_terms, f, ensure_ascii=False, indent=2)
    
    print(f"Done! Output saved to {output_file}")
    
    return cui_terms


def main():
    parser = argparse.ArgumentParser(
        description="Extract preferred terms from UMLS MRCONSO.RRF file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_preferred_terms.py
  python extract_preferred_terms.py --language SPA  # Spanish terms
  python extract_preferred_terms.py --input /path/to/MRCONSO.RRF --output terms.json

Common language codes:
  ENG - English (default)
  SPA - Spanish
  FRE - French
  GER - German
  POR - Portuguese
  ITA - Italian
  DUT - Dutch
  JPN - Japanese
  KOR - Korean
  CHI - Chinese
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
        default=PREFERRED_TERMS_OUTPUT,
        help=f"Path for output JSON file (default: {PREFERRED_TERMS_OUTPUT})"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default='ENG',
        help="Language code to filter by (default: ENG)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_directories()
    
    # Validate input file
    if not validate_input_file(args.input, "MRCONSO.RRF"):
        sys.exit(1)
    
    # Run extraction
    extract_preferred_terms(args.input, args.output, args.language)


if __name__ == "__main__":
    main()
