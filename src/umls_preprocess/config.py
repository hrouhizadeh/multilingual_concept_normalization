#!/usr/bin/env python3
"""
UMLS Preprocessing Configuration

This module provides centralized path configuration for all UMLS preprocessing scripts.
Users should modify the paths below to match their local setup.
"""

from pathlib import Path

# =============================================================================
# BASE DIRECTORIES - MODIFY THESE TO MATCH YOUR SETUP
# =============================================================================

# Root directory of this project (automatically detected)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Directory containing UMLS RRF files
# Download UMLS from: https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
# After extraction, point this to the META folder containing the .RRF files
UMLS_META_DIR = PROJECT_ROOT / "data" / "umls" / "META"

# Directory for output JSON files
OUTPUT_DIR = PROJECT_ROOT / "output"

# =============================================================================
# INPUT FILES (UMLS RRF FILES)
# =============================================================================

# MRCONSO.RRF - Concept names and sources
# Contains all concept names, their sources, and language information
MRCONSO_FILE = UMLS_META_DIR / "MRCONSO.RRF"

# MRDEF.RRF - Definitions
# Contains definitions for concepts from various source vocabularies
MRDEF_FILE = UMLS_META_DIR / "MRDEF.RRF"

# MRREL.RRF - Relationships
# Contains relationships between concepts (parent/child, broader/narrower, etc.)
MRREL_FILE = UMLS_META_DIR / "MRREL.RRF"

# MRSTY.RRF - Semantic Types
# Maps concepts to their semantic types (e.g., Disease, Drug, Procedure)
MRSTY_FILE = UMLS_META_DIR / "MRSTY.RRF"

# SemGroups.txt - Semantic Groups
# Maps semantic type codes (T-codes) to semantic groups
SEMGROUPS_FILE = UMLS_META_DIR / "SemGroups.txt"

# =============================================================================
# OUTPUT FILES
# =============================================================================

# Definitions output: {CUI: {SOURCE: definition}}
DEFINITIONS_OUTPUT = OUTPUT_DIR / "umls_definitions.json"

# Semantic types output: {CUI: {sem_T_code: [...], sem_type: [...], sem_group: [...]}}
SEMANTIC_TYPES_OUTPUT = OUTPUT_DIR / "cui_semantic_mapping.json"

# Hierarchies output: {CUI: {hypernyms: [...], hyponyms: [...]}}
HIERARCHIES_OUTPUT = OUTPUT_DIR / "umls_hierarchies.json"

# Preferred terms output: {CUI: preferred_term}
PREFERRED_TERMS_OUTPUT = OUTPUT_DIR / "umls_preferred_terms.json"

# All terms output: JSONL file with all terms per CUI
ALL_TERMS_OUTPUT = OUTPUT_DIR / "umls_all_terms.jsonl"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UMLS_META_DIR.mkdir(parents=True, exist_ok=True)


def validate_input_file(filepath: Path, file_description: str) -> bool:
    """
    Check if an input file exists and print helpful message if not.
    
    Args:
        filepath: Path to the file to check
        file_description: Human-readable description of the file
    
    Returns:
        True if file exists, False otherwise
    """
    if not filepath.exists():
        print(f"ERROR: {file_description} not found at:")
        print(f"       {filepath}")
        print(f"\nPlease ensure you have:")
        print(f"  1. Downloaded UMLS from https://www.nlm.nih.gov/research/umls/")
        print(f"  2. Extracted the files to: {UMLS_META_DIR}")
        print(f"  3. Or updated the paths in config.py")
        return False
    return True


def get_config_summary() -> str:
    """Return a summary of current configuration for display."""
    return f"""
UMLS Preprocessing Configuration
================================
Project Root:     {PROJECT_ROOT}
UMLS META Dir:    {UMLS_META_DIR}
Output Dir:       {OUTPUT_DIR}

Input Files:
  MRCONSO.RRF:    {MRCONSO_FILE} {'✓' if MRCONSO_FILE.exists() else '✗'}
  MRDEF.RRF:      {MRDEF_FILE} {'✓' if MRDEF_FILE.exists() else '✗'}
  MRREL.RRF:      {MRREL_FILE} {'✓' if MRREL_FILE.exists() else '✗'}
  MRSTY.RRF:      {MRSTY_FILE} {'✓' if MRSTY_FILE.exists() else '✗'}
  SemGroups.txt:  {SEMGROUPS_FILE} {'✓' if SEMGROUPS_FILE.exists() else '✗'}
"""


if __name__ == "__main__":
    print(get_config_summary())
