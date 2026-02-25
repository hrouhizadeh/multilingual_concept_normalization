#!/usr/bin/env python3
"""
Dataset Preprocessing - Run All Scripts

Runs all dataset preprocessing scripts in sequence.

Usage:
    python run_all.py              # Run all scripts
    python run_all.py --check      # Only check if input files exist
    python run_all.py --list       # List available datasets
    python run_all.py --only bc5cdr n2c2  # Run specific datasets only
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import config


# Dataset configurations: (script_name, dataset_dir, description)
DATASETS = {
    'bc5cdr': (
        'scripts/preprocess_bc5cdr.py',
        config.BC5CDR_DIR,
        'BC5CDR - English disease/chemical NER (MeSH)'
    ),
    'n2c2': (
        'scripts/preprocess_n2c2.py',
        config.N2C2_DIR,
        'N2C2 - English clinical NLP (RxNorm, SNOMED)'
    ),
    'quaero': (
        'scripts/preprocess_quaero.py',
        config.QUAERO_DIR,
        'Quaero - French medical corpus (UMLS)'
    ),
    'meducd': (
        'scripts/preprocess_meducd.py',
        config.MED_UCD_DIR,
        'MedUCD - French medication (ATC)'
    ),
    'distemist': (
        'scripts/preprocess_distemist.py',
        config.DISTEMIST_DIR,
        'DisTEMIST - Spanish diseases (SNOMED)'
    ),
    'pharma': (
        'scripts/preprocess_pharma.py',
        config.PHARMA_DIR,
        'Pharma - Spanish pharmaceutical (SNOMED)'
    ),
    'bronco': (
        'scripts/preprocess_bronco.py',
        config.BRONCO_DIR,
        'BRONCO - German clinical (ATC, ICD-10)'
    ),
    'tllv': (
        'scripts/preprocess_tllv.py',
        config.TLLV_DIR,
        'Turkish LOINC - Lab tests (LOINC)'
    ),
    'gsc': (
        'scripts/preprocess_gsc.py',
        config.GSC_DIR,
        'GSC/MANTRA - Multilingual (MedDRA, MeSH, SNOMED)'
    ),
    'xlbel': (
        'scripts/preprocess_xlbel.py',
        config.XL_BEL_DIR,
        'XL-BEL - Multilingual entity linking (UMLS)'
    ),
}


def check_dataset(name: str, dataset_dir: Path) -> bool:
    """Check if dataset directory exists."""
    exists = dataset_dir.exists()
    status = '✓' if exists else '✗'
    print(f"  {status} {name}: {dataset_dir}")
    return exists


def check_umls_files() -> bool:
    """Check if required UMLS files exist."""
    print("\nChecking UMLS files...")
    
    files = [
        (config.SEMANTIC_MAPPING_FILE, "Semantic mapping"),
        (config.WORD_ONTOLOGY_MAP_FILE, "Word-ontology map"),
        (config.CUI_CODES_FILE, "CUI codes"),
    ]
    
    all_ok = True
    for path, name in files:
        exists = path.exists()
        status = '✓' if exists else '✗'
        print(f"  {status} {name}: {path}")
        if not exists:
            all_ok = False
    
    return all_ok


def run_script(script_path: str, description: str) -> bool:
    """Run a single preprocessing script."""
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
        print(f"\n✗ Script failed with return code {result.returncode}")
        return False
    
    print(f"\n✓ Completed: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run dataset preprocessing scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  bc5cdr    - BC5CDR (English, MeSH)
  n2c2      - N2C2 (English, RxNorm/SNOMED)
  quaero    - Quaero (French, UMLS)
  meducd    - MedUCD (French, ATC)
  distemist - DisTEMIST (Spanish, SNOMED)
  pharma    - Pharma (Spanish, SNOMED)
  bronco    - BRONCO (German, ATC/ICD-10)
  tllv      - TLLV LOINC (Turkish, LOINC)
  gsc       - GSC/MANTRA (Multilingual)
  xlbel     - XL-BEL (Multilingual)
        """
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if input files exist, don't run scripts"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit"
    )
    parser.add_argument(
        "--only",
        nargs='+',
        choices=list(DATASETS.keys()),
        help="Only process specified datasets"
    )
    parser.add_argument(
        "--skip",
        nargs='+',
        choices=list(DATASETS.keys()),
        help="Skip specified datasets"
    )
    
    args = parser.parse_args()
    
    # List datasets
    if args.list:
        print("Available datasets:")
        for name, (_, _, desc) in DATASETS.items():
            print(f"  {name:12} - {desc}")
        return
    
    # Determine which datasets to process
    datasets_to_run = list(DATASETS.keys())
    
    if args.only:
        datasets_to_run = args.only
    
    if args.skip:
        datasets_to_run = [d for d in datasets_to_run if d not in args.skip]
    
    # Ensure output directories exist
    config.ensure_directories()
    
    # Check UMLS files
    umls_ok = check_umls_files()
    
    # Check dataset directories
    print("\nChecking dataset directories...")
    available_datasets = []
    for name in datasets_to_run:
        script, dataset_dir, desc = DATASETS[name]
        if check_dataset(name, dataset_dir):
            available_datasets.append(name)
    
    if not umls_ok:
        print("\n⚠ Missing UMLS files. Please ensure UMLS data is processed first.")
        print("  Run the umls_preprocessing scripts or update paths in config.py")
    
    if not available_datasets:
        print("\n⚠ No dataset directories found. Please:")
        print("  1. Download the datasets")
        print("  2. Place them in the data/datasets/ directory")
        print("  3. Update paths in config.py if needed")
        sys.exit(1)
    
    if args.check:
        print(f"\nFile check complete. {len(available_datasets)} datasets available.")
        print("Use 'python run_all.py' to run preprocessing.")
        sys.exit(0 if umls_ok else 1)
    
    if not umls_ok:
        print("\nCannot proceed without UMLS files.")
        sys.exit(1)
    
    # Run preprocessing scripts
    print("\n" + "="*60)
    print("Starting Dataset Preprocessing Pipeline")
    print("="*60)
    print(f"Datasets to process: {', '.join(available_datasets)}")
    
    successful = 0
    failed = 0
    skipped = 0
    
    for name in datasets_to_run:
        script, dataset_dir, desc = DATASETS[name]
        
        if name not in available_datasets:
            print(f"\n⚠ Skipping {name} - directory not found")
            skipped += 1
            continue
        
        if run_script(script, desc):
            successful += 1
        else:
            failed += 1
            print(f"\n⚠ Continuing to next dataset despite failure...")
    
    # Summary
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed:     {failed}")
    print(f"  ⊘ Skipped:    {skipped}")
    
    if successful > 0:
        print(f"\nOutput files saved to:")
        print(f"  By dataset: {config.BY_DATASET_OUTPUT_DIR}")
        print(f"  Combined:   {config.ALL_DATASETS_OUTPUT_DIR}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
