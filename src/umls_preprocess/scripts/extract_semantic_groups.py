#!/usr/bin/env python3
"""
UMLS semantic group/type extractor

Creates a JSON file mapping UMLS CUIs to semantic types and groups.

Input files:
    1. SemGroups.txt - Maps T-codes to semantic types and groups
       Format: ACTI|Activities & Behaviors|T052|Activity

    2. MRSTY.RRF - Maps CUIs to T-codes
       Format: C0000005|T116|A1.4.1.2.1.7|Amino Acid, Peptide, or Protein|AT17648347||

Output format:
    {
        "C0028077": {
            "sem_T_code": ["T047"],
            "sem_type": ["Disease or Syndrome"],
            "sem_group": ["Disorders"]
        },
        ...
    }

Usage:
    python extract_semantic_types.py
    python extract_semantic_types.py --mrsty /path/to/MRSTY.RRF --semgroups /path/to/SemGroups.txt
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MRSTY_FILE,
    SEMGROUPS_FILE,
    SEMANTIC_TYPES_OUTPUT,
    ensure_directories,
    validate_input_file,
)


def parse_semgroups(semgroups_file: Path) -> dict:
    """
    Parse SemGroups.txt to create a mapping from T-code to semantic type and group.
    
    Args:
        semgroups_file: Path to SemGroups.txt
    
    Returns:
        dict: {T-code: {"sem_type": str, "sem_group": str}}
    """
    tcode_mapping = {}
    
    with open(semgroups_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) >= 4:
                sem_group_name = parts[1]   # e.g., "Activities & Behaviors"
                t_code = parts[2]           # e.g., "T052"
                sem_type = parts[3]         # e.g., "Activity"
                
                tcode_mapping[t_code] = {
                    "sem_type": sem_type,
                    "sem_group": sem_group_name
                }
    
    return tcode_mapping


def parse_mrsty(mrsty_file: Path, tcode_mapping: dict) -> dict:
    """
    Parse MRSTY.RRF to create the final CUI mapping.
    
    Args:
        mrsty_file: Path to MRSTY.RRF
        tcode_mapping: Dict from parse_semgroups()
    
    Returns:
        dict: {CUI: {"sem_T_code": [...], "sem_type": [...], "sem_group": [...]}}
    """
    cui_data = defaultdict(lambda: {
        "sem_T_code": [],
        "sem_type": [],
        "sem_group": []
    })
    
    line_count = 0
    with open(mrsty_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) >= 2:
                cui = parts[0]      # e.g., "C0000005"
                t_code = parts[1]   # e.g., "T116"
                
                # Only add if we haven't already added this T-code for this CUI
                if t_code not in cui_data[cui]["sem_T_code"]:
                    cui_data[cui]["sem_T_code"].append(t_code)
                    
                    # Look up the semantic type and group
                    if t_code in tcode_mapping:
                        sem_type = tcode_mapping[t_code]["sem_type"]
                        sem_group = tcode_mapping[t_code]["sem_group"]
                        
                        if sem_type not in cui_data[cui]["sem_type"]:
                            cui_data[cui]["sem_type"].append(sem_type)
                        if sem_group not in cui_data[cui]["sem_group"]:
                            cui_data[cui]["sem_group"].append(sem_group)
            
            line_count += 1
            if line_count % 500000 == 0:
                print(f"  Processed {line_count:,} lines...")
    
    # Convert defaultdict to regular dict
    return dict(cui_data)


def main():
    parser = argparse.ArgumentParser(
        description="Extract semantic types from UMLS MRSTY.RRF and SemGroups.txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_semantic_types.py
  python extract_semantic_types.py --mrsty /path/to/MRSTY.RRF --semgroups /path/to/SemGroups.txt
        """
    )
    parser.add_argument(
        "--mrsty",
        type=Path,
        default=MRSTY_FILE,
        help=f"Path to MRSTY.RRF file (default: {MRSTY_FILE})"
    )
    parser.add_argument(
        "--semgroups",
        type=Path,
        default=SEMGROUPS_FILE,
        help=f"Path to SemGroups.txt file (default: {SEMGROUPS_FILE})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=SEMANTIC_TYPES_OUTPUT,
        help=f"Path for output JSON file (default: {SEMANTIC_TYPES_OUTPUT})"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_directories()
    
    # Validate input files
    if not validate_input_file(args.semgroups, "SemGroups.txt"):
        sys.exit(1)
    if not validate_input_file(args.mrsty, "MRSTY.RRF"):
        sys.exit(1)
    
    # Parse SemGroups.txt
    print("Parsing SemGroups.txt...")
    tcode_mapping = parse_semgroups(args.semgroups)
    print(f"  Found {len(tcode_mapping)} T-code mappings")
    
    # Parse MRSTY.RRF
    print(f"Parsing {args.mrsty.name}...")
    cui_mapping = parse_mrsty(args.mrsty, tcode_mapping)
    print(f"  Found {len(cui_mapping):,} CUI mappings")
    
    # Write output
    print(f"Writing output to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(cui_mapping, f, indent=2, ensure_ascii=False)
    
    print("Done!")
    
    # Print sample entries
    print("\nSample entries:")
    for i, (cui, data) in enumerate(cui_mapping.items()):
        if i >= 3:
            break
        print(f'  "{cui}": {json.dumps(data)}')


if __name__ == "__main__":
    main()
