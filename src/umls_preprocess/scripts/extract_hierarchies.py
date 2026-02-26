#!/usr/bin/env python3
"""
UMLS hierarchy extractor

Extracts hypernym/hyponym (parent/child) relationships from UMLS MRREL.RRF file.

MRREL.RRF Format (pipe-delimited):
    Column 0: CUI1 - Concept Unique Identifier 1
    Column 1: AUI1 - Atom Unique Identifier 1
    Column 2: STYPE1 - Source Type 1
    Column 3: REL - Relationship
    Column 4: CUI2 - Concept Unique Identifier 2
    ...

Relationship types used:
    - PAR (Parent of) / CHD (Child of): Direct parent-child relationships
    - RB (Broader) / RN (Narrower): Broader/narrower relationships

Output format:
    {
        "C0000039": {
            "hypernyms": ["C0001234", "C0005678"],
            "hyponyms": ["C0009012", "C0003456"]
        },
        ...
    }

Usage:
    python extract_hierarchies.py
    python extract_hierarchies.py --input /path/to/MRREL.RRF --output hierarchies.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MRREL_FILE,
    HIERARCHIES_OUTPUT,
    ensure_directories,
    validate_input_file,
)


def extract_all_hierarchies(mrrel_path: Path, output_json_path: Path) -> dict:
    """
    Extract all hypernym/hyponym relationships for all CUIs from MRREL.RRF
    and save to JSON file.
    
    Args:
        mrrel_path: Path to MRREL.RRF file
        output_json_path: Path for output JSON file
    
    Returns:
        Dictionary of hierarchies
    """
    hierarchies = {}
    
    # Relationship types indicating hypernyms (parents/broader concepts)
    hypernym_rels = {'PAR', 'RB'}
    # Relationship types indicating hyponyms (children/narrower concepts)
    hyponym_rels = {'CHD', 'RN'}
    
    print(f"Reading {mrrel_path}...")
    
    line_count = 0
    with open(mrrel_path, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split('|')
            
            if len(fields) < 5:
                continue
            
            cui1 = fields[0]
            rel = fields[3]
            cui2 = fields[4]
            
            # Skip if not a hierarchical relationship
            if rel not in hypernym_rels and rel not in hyponym_rels:
                continue
            
            # Initialize if not exists
            if cui1 not in hierarchies:
                hierarchies[cui1] = {'hypernyms': [], 'hyponyms': []}
            if cui2 not in hierarchies:
                hierarchies[cui2] = {'hypernyms': [], 'hyponyms': []}
            
            # Add relationships
            if rel in hypernym_rels:
                # CUI1 has CUI2 as hypernym (parent)
                hierarchies[cui1]['hypernyms'].append(cui2)
                # CUI2 has CUI1 as hyponym (child)
                hierarchies[cui2]['hyponyms'].append(cui1)
            elif rel in hyponym_rels:
                # CUI1 has CUI2 as hyponym (child)
                hierarchies[cui1]['hyponyms'].append(cui2)
                # CUI2 has CUI1 as hypernym (parent)
                hierarchies[cui2]['hypernyms'].append(cui1)
            
            line_count += 1
            if line_count % 1000000 == 0:
                print(f"  Processed {line_count:,} relationship lines...")
    
    print(f"Processed {line_count:,} hierarchical relationships")
    print("Removing duplicates...")
    
    # Remove duplicates
    for cui in hierarchies:
        hierarchies[cui]['hypernyms'] = list(set(hierarchies[cui]['hypernyms']))
        hierarchies[cui]['hyponyms'] = list(set(hierarchies[cui]['hyponyms']))
    
    print(f"Writing to {output_json_path}...")
    
    # Save to JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchies, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Total CUIs with hierarchy data: {len(hierarchies):,}")
    
    return hierarchies


def main():
    parser = argparse.ArgumentParser(
        description="Extract hierarchical relationships from UMLS MRREL.RRF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_hierarchies.py
  python extract_hierarchies.py --input /path/to/MRREL.RRF --output hierarchies.json

Relationship types:
  - PAR/CHD: Parent/Child relationships
  - RB/RN: Broader/Narrower relationships
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=MRREL_FILE,
        help=f"Path to MRREL.RRF file (default: {MRREL_FILE})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=HIERARCHIES_OUTPUT,
        help=f"Path for output JSON file (default: {HIERARCHIES_OUTPUT})"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_directories()
    
    # Validate input file
    if not validate_input_file(args.input, "MRREL.RRF"):
        sys.exit(1)
    
    # Run extraction
    extract_all_hierarchies(args.input, args.output)


if __name__ == "__main__":
    main()
