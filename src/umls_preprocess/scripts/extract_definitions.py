#!/usr/bin/env python3
"""
UMLSd efinition extractor

Reads the UMLS MRDEF.RRF file and creates a JSON file grouped by CUI and source vocabulary.

MRDEF.RRF Format (pipe-delimited):
    Column 0: CUI  - Concept Unique Identifier
    Column 1: AUI  - Atom Unique Identifier
    Column 2: ATUI - Attribute Unique Identifier
    Column 3: SATUI - Source Attribute Unique Identifier
    Column 4: SAB  - Source Abbreviation (e.g., MSH, SNOMEDCT_US)
    Column 5: DEF  - Definition text
    Column 6: SUPPRESS - Suppressibility flag
    Column 7: CVF  - Content View Flag (usually empty)

Output format:
    [
        {"CUI1": {"SOURCE1": "definition1", "SOURCE2": "definition2"}},
        {"CUI2": {"SOURCE1": "definition1"}}
    ]

Usage:
    python extract_definitions.py
    python extract_definitions.py --input /path/to/MRDEF.RRF --output /path/to/output.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MRDEF_FILE,
    DEFINITIONS_OUTPUT,
    ensure_directories,
    validate_input_file,
)


def parse_mrdef(input_file: Path, output_file: Path) -> list:
    """
    Parse MRDEF.RRF file and convert to JSON format.
    
    Args:
        input_file: Path to MRDEF.RRF file
        output_file: Path for output JSON file
    
    Returns:
        List of dictionaries in format [{CUI: {SAB: DEF}}]
    """
    # Use defaultdict to group by CUI, then by source
    cui_definitions = defaultdict(dict)
    
    print(f"Reading {input_file}...")
    
    line_count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            
            # Split by pipe delimiter
            parts = line.split('|')
            
            if len(parts) < 6:
                print(f"Warning: Skipping malformed line {line_count + 1}: {line[:50]}...")
                continue
            
            cui = parts[0]          # Concept Unique Identifier
            sab = parts[4]          # Source Abbreviation
            definition = parts[5]   # Definition text
            
            # Store definition by CUI and source
            # If same CUI+SAB combo exists, keep the first one
            if sab not in cui_definitions[cui]:
                cui_definitions[cui][sab] = definition
            
            line_count += 1
            
            # Progress indicator for large files
            if line_count % 100000 == 0:
                print(f"  Processed {line_count:,} lines...")
    
    print(f"Finished processing {line_count:,} lines")
    print(f"Found {len(cui_definitions):,} unique CUIs")
    
    # Convert to the requested output format: [{CUI: {SAB: DEF}}]
    result = [{cui: sources} for cui, sources in cui_definitions.items()]
    
    # Write JSON output
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Done! Output saved to {output_file}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract definitions from UMLS MRDEF.RRF file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_definitions.py
  python extract_definitions.py --input /path/to/MRDEF.RRF --output definitions.json
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=MRDEF_FILE,
        help=f"Path to MRDEF.RRF file (default: {MRDEF_FILE})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFINITIONS_OUTPUT,
        help=f"Path for output JSON file (default: {DEFINITIONS_OUTPUT})"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_directories()
    
    # Validate input file
    if not validate_input_file(args.input, "MRDEF.RRF"):
        sys.exit(1)
    
    # Run extraction
    parse_mrdef(args.input, args.output)


if __name__ == "__main__":
    main()
