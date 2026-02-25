#!/usr/bin/env python3
"""
UMLS Preprocessing - Run All Scripts

This script runs all UMLS preprocessing scripts in sequence.
It's useful for generating all JSON outputs at once.

Usage:
    python run_all.py           # Run all scripts
    python run_all.py --check   # Only check if input files exist
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add current directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MRCONSO_FILE,
    MRDEF_FILE,
    MRREL_FILE,
    MRSTY_FILE,
    SEMGROUPS_FILE,
    ensure_directories,
    get_config_summary,
)


# Scripts to run in order (script name, required files, description)
SCRIPTS = [
    (
        "scripts/extract_definitions.py",
        [MRDEF_FILE],
        "Extract definitions from MRDEF.RRF"
    ),
    (
        "scripts/extract_semantic_types.py",
        [MRSTY_FILE, SEMGROUPS_FILE],
        "Extract semantic types from MRSTY.RRF"
    ),
    (
        "scripts/extract_hierarchies.py",
        [MRREL_FILE],
        "Extract hierarchies from MRREL.RRF"
    ),
    (
        "scripts/extract_preferred_terms.py",
        [MRCONSO_FILE],
        "Extract preferred terms from MRCONSO.RRF"
    ),
    (
        "scripts/extract_all_terms.py",
        [MRCONSO_FILE],
        "Extract all terms from MRCONSO.RRF (memory intensive)"
    ),
]


def check_files() -> bool:
    """Check if all required input files exist."""
    print("Checking input files...")
    print(get_config_summary())
    
    all_files = set()
    for _, files, _ in SCRIPTS:
        all_files.update(files)
    
    missing = [f for f in all_files if not f.exists()]
    
    if missing:
        print("\n❌ Missing files:")
        for f in missing:
            print(f"   - {f}")
        print("\nPlease download UMLS and place RRF files in the data/umls/META/ directory.")
        return False
    
    print("✓ All input files found!")
    return True


def run_script(script_path: str, description: str) -> bool:
    """Run a single script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script:  {script_path}")
    print('='*60)
    
    script_full_path = Path(__file__).parent / script_path
    
    result = subprocess.run(
        [sys.executable, str(script_full_path)],
        cwd=str(Path(__file__).parent)
    )
    
    if result.returncode != 0:
        print(f"\n❌ Script failed with return code {result.returncode}")
        return False
    
    print(f"\n✓ Completed: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run all UMLS preprocessing scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This will run the following scripts in order:
  1. extract_definitions.py     - Extract definitions from MRDEF.RRF
  2. extract_semantic_types.py  - Extract semantic types from MRSTY.RRF
  3. extract_hierarchies.py     - Extract hierarchies from MRREL.RRF
  4. extract_preferred_terms.py - Extract preferred terms from MRCONSO.RRF
  5. extract_all_terms.py       - Extract all terms from MRCONSO.RRF
        """
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if input files exist, don't run scripts"
    )
    parser.add_argument(
        "--skip-all-terms",
        action="store_true",
        help="Skip extract_all_terms.py (memory intensive)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_directories()
    
    # Check files
    if not check_files():
        sys.exit(1)
    
    if args.check:
        print("\nFile check passed. Use 'python run_all.py' to run all scripts.")
        sys.exit(0)
    
    # Run all scripts
    print("\n" + "="*60)
    print("Starting UMLS Preprocessing Pipeline")
    print("="*60)
    
    scripts_to_run = SCRIPTS
    if args.skip_all_terms:
        scripts_to_run = [s for s in SCRIPTS if "extract_all_terms" not in s[0]]
        print("\nNote: Skipping extract_all_terms.py (--skip-all-terms flag)")
    
    successful = 0
    failed = 0
    
    for script_path, required_files, description in scripts_to_run:
        # Check if required files exist
        missing = [f for f in required_files if not f.exists()]
        if missing:
            print(f"\n⚠ Skipping {script_path} - missing required files:")
            for f in missing:
                print(f"   - {f}")
            continue
        
        if run_script(script_path, description):
            successful += 1
        else:
            failed += 1
            print(f"\n⚠ Continuing to next script despite failure...")
    
    # Summary
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed:     {failed}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
