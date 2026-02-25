#!/usr/bin/env python3
"""
Shared Utility Functions for Dataset Preprocessing

Contains common functions used across all dataset preprocessing scripts:
- JSON file reading/writing
- CSV file creation
- Semantic type/group extraction
- Exact match status calculation
- File discovery utilities
- List flattening
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from sklearn.model_selection import train_test_split


# =============================================================================
# FILE I/O UTILITIES
# =============================================================================

def read_json_file(file_path: Union[str, Path]) -> Any:
    """
    Read and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON content (dict, list, etc.)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(file_path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """
    Write data to a JSON file.
    
    Args:
        file_path: Path to output JSON file
        data: Data to write
        indent: JSON indentation level
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def create_csv_file(
    output_file_path: Union[str, Path],
    content: List[List[Any]],
    language: str,
    source: str,
    target_ontologies: List[str],
    columns: Optional[List[str]] = None
) -> int:
    """
    Create a CSV file from processed content with deduplication.
    
    Args:
        output_file_path: Path for output CSV file
        content: List of rows, each row is [term, code, sem_type, sem_group, em_stat]
        language: Language code (e.g., 'eng', 'fre', 'spa')
        source: Dataset source name (e.g., 'bc5cdr', 'quaero')
        target_ontologies: List of target ontologies
        columns: Optional custom column names
        
    Returns:
        Number of unique rows written
    """
    if columns is None:
        columns = ['term', 'code', 'langauge', 'semantic_type', 'semantic_group',
                   'targe_ontologies', 'exact_match', 'source']
    
    # Ensure parent directory exists
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    seen_terms_lower = set()
    rows_written = 0
    
    with open(output_file_path, 'w', newline='', encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(columns)
        
        for row in content:
            term = row[0]
            code = row[1]
            s_type = row[2]
            s_group = row[3]
            em_stat = row[4]
            
            # Skip rows with empty semantic info if needed
            if s_group == [] or s_type == []:
                continue
                
            term_lower = str(term).lower()
            if term_lower in seen_terms_lower:
                continue
            
            seen_terms_lower.add(term_lower)
            csv_writer.writerow([
                term, code, language, s_type, s_group,
                target_ontologies, em_stat, source
            ])
            rows_written += 1
    
    print(f"Successfully created CSV file: {output_file_path} ({rows_written} rows)")
    return rows_written


def create_csv_file_dict(
    output_file_path: Union[str, Path],
    instances: Dict[str, tuple],
    language: str,
    source: str,
    target_ontologies: List[str]
) -> int:
    """
    Create CSV file from dictionary of instances (used by BRONCO).
    
    Args:
        output_file_path: Path for output CSV file
        instances: Dict mapping ID to (term, codes, sem_types, sem_groups, em_stat)
        language: Language code
        source: Dataset source name
        target_ontologies: List of target ontologies
        
    Returns:
        Number of unique rows written
    """
    columns = ['term', 'code', 'langauge', 'semantic_type', 'semantic_group',
               'targe_ontologies', 'exact_match', 'source']
    
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    seen_terms_lower = set()
    rows_written = 0
    
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        
        for _, (term, codes, s_typ, s_gro, ex_m) in instances.items():
            term_lower = str(term).lower()
            if term_lower in seen_terms_lower:
                continue
            
            seen_terms_lower.add(term_lower)
            writer.writerow({
                'term': term,
                'code': codes,
                'langauge': language,
                'semantic_type': s_typ,
                'semantic_group': s_gro,
                'targe_ontologies': target_ontologies,
                'exact_match': ex_m,
                'source': source
            })
            rows_written += 1
    
    print(f"Successfully created CSV file: {output_file_path} ({rows_written} rows)")
    return rows_written


def create_csv_file_multilang(
    output_file_path: Union[str, Path],
    data: List[List[Any]]
) -> int:
    """
    Create CSV file for multilingual data (GSC, XL-BEL format).
    
    Args:
        output_file_path: Path for output CSV file
        data: List of rows with full column data including language
        
    Returns:
        Number of rows written
    """
    columns = ['term', 'code', 'langauge', 'semantic_type', 'semantic_group',
               'targe_ontologies', 'exact_match', 'source']
    
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file_path, 'w', newline='', encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(columns)
        csv_writer.writerows(data)
    
    print(f"Successfully created CSV file: {output_file_path} ({len(data)} rows)")
    return len(data)


# =============================================================================
# SEMANTIC TYPE/GROUP EXTRACTION
# =============================================================================

def extract_semantic_types_groups(
    codes: List[str],
    semantic_mapping: Dict[str, Dict]
) -> tuple:
    """
    Extract semantic types and groups for given CUI codes.
    
    Args:
        codes: List of UMLS CUI codes
        semantic_mapping: Dict mapping CUI to {sem_type: [...], sem_group: [...]}
        
    Returns:
        Tuple of (unique_sem_types, unique_sem_groups)
    """
    s_groups = []
    s_types = []
    
    for cui in codes:
        if cui not in semantic_mapping:
            continue
        s_groups.append(semantic_mapping[cui].get('sem_group', []))
        s_types.append(semantic_mapping[cui].get('sem_type', []))
    
    # Flatten nested lists
    s_types = flatten_list(s_types)
    s_groups = flatten_list(s_groups)
    
    # Remove duplicates while preserving order
    s_types = list(dict.fromkeys(s_types))
    s_groups = list(dict.fromkeys(s_groups))
    
    return s_types, s_groups


# =============================================================================
# EXACT MATCH STATUS CALCULATION
# =============================================================================

def calculate_exact_match_status(
    term: str,
    word_ontology_map: Dict[str, List[str]],
    train_dev_terms: Set[str],
    ontology_prefixes: List[str]
) -> int:
    """
    Calculate exact match status for a term.
    
    Status codes:
        0 - Not in train/dev and not in UMLS with target ontology
        1 - In UMLS with target ontology only
        2 - In train/dev only
        3 - In both train/dev and UMLS with target ontology
    
    Args:
        term: The term to check
        word_ontology_map: Dict mapping terms to their source vocabularies
        train_dev_terms: Set of terms in training/dev sets
        ontology_prefixes: List of ontology prefixes to check (e.g., ['MSH', 'SNO'])
        
    Returns:
        Integer status code (0-3)
    """
    term_lower = term.lower()
    mappings = word_ontology_map.get(term_lower, [])
    
    in_train_dev = term_lower in train_dev_terms
    in_umls = any(
        any(m.startswith(prefix) for prefix in ontology_prefixes)
        for m in mappings
    )
    
    if in_train_dev and in_umls:
        return 3
    elif in_train_dev:
        return 2
    elif in_umls:
        return 1
    return 0


def calculate_exact_match_simple(
    term: str,
    word_ontology_map: Dict[str, List[str]]
) -> int:
    """
    Simple exact match status (for test-only datasets like GSC, XL-BEL).
    
    Args:
        term: The term to check
        word_ontology_map: Dict mapping terms to their source vocabularies
        
    Returns:
        1 if term found in UMLS, 0 otherwise
    """
    term_lower = term.lower()
    return 1 if word_ontology_map.get(term_lower) else 0


# =============================================================================
# FILE DISCOVERY UTILITIES
# =============================================================================

def find_files_by_extension(
    directory: Union[str, Path],
    extension: str,
    recursive: bool = False
) -> List[str]:
    """
    Find all files with given extension in a directory.
    
    Args:
        directory: Directory to search
        extension: File extension (e.g., '.ann', '.tsv', '.xml')
        recursive: Whether to search subdirectories
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Warning: Directory {directory} does not exist.")
        return []
    
    if recursive:
        return [str(p) for p in directory.rglob(f"*{extension}")]
    else:
        return [str(p) for p in directory.glob(f"*{extension}")]


def find_ann_files(directory: Union[str, Path], recursive: bool = True) -> List[str]:
    """Find all .ann files in directory."""
    return find_files_by_extension(directory, '.ann', recursive)


def find_tsv_files(directory: Union[str, Path]) -> List[str]:
    """Find all .tsv files in directory."""
    return find_files_by_extension(directory, '.tsv', recursive=False)


def find_xml_files(directory: Union[str, Path]) -> List[str]:
    """Find all .xml files in directory."""
    return find_files_by_extension(directory, '.xml', recursive=False)


# =============================================================================
# LIST UTILITIES
# =============================================================================

def flatten_list(nested_list: List) -> List:
    """
    Recursively flatten a nested list.
    
    Args:
        nested_list: A potentially nested list
        
    Returns:
        Flattened list
    """
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            flat_list.extend(flatten_list(element))
        else:
            flat_list.append(element)
    return flat_list


# =============================================================================
# DATA SPLITTING UTILITIES
# =============================================================================

def split_train_dev(
    data: List,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Split data into training and development sets.
    
    Args:
        data: List of data items
        test_size: Fraction for dev set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_set, dev_set)
    """
    return train_test_split(data, test_size=test_size, random_state=random_state)


def split_train_dev_test(
    data: List,
    train_size: float = 0.6,
    dev_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Split data into train, dev, and test sets.
    
    Args:
        data: List of data items
        train_size: Fraction for training set
        dev_size: Fraction for dev set (test gets remainder)
        random_state: Random seed
        
    Returns:
        Tuple of (train_set, dev_set, test_set)
    """
    # First split: train vs (dev + test)
    train_set, dev_test = train_test_split(
        data, 
        test_size=(1 - train_size),
        random_state=random_state
    )
    
    # Second split: dev vs test
    relative_dev_size = dev_size / (1 - train_size)
    dev_set, test_set = train_test_split(
        dev_test,
        test_size=(1 - relative_dev_size),
        random_state=random_state
    )
    
    return train_set, dev_set, test_set


# =============================================================================
# UMLS DATA LOADERS
# =============================================================================

class UMLSDataLoader:
    """
    Helper class to load and cache UMLS data files.
    """
    
    def __init__(self, config):
        """
        Initialize with config module.
        
        Args:
            config: The config module with file paths
        """
        self.config = config
        self._semantic_mapping = None
        self._word_ontology_map = None
        self._cui_codes = None
        self._mapping_files = {}
    
    @property
    def semantic_mapping(self) -> Dict:
        """Load and cache semantic mapping."""
        if self._semantic_mapping is None:
            self._semantic_mapping = read_json_file(self.config.SEMANTIC_MAPPING_FILE)
        return self._semantic_mapping
    
    @property
    def word_ontology_map(self) -> Dict:
        """Load and cache word-to-ontology mapping (lowercased keys)."""
        if self._word_ontology_map is None:
            data = read_json_file(self.config.WORD_ONTOLOGY_MAP_FILE)
            self._word_ontology_map = {k.lower(): v for k, v in data.items()}
        return self._word_ontology_map
    
    @property
    def cui_codes(self) -> List:
        """Load and cache CUI codes list."""
        if self._cui_codes is None:
            self._cui_codes = read_json_file(self.config.CUI_CODES_FILE)
        return self._cui_codes
    
    def get_mapping(self, mapping_name: str) -> Dict:
        """
        Load a specific mapping file (mesh, snomed, atc, icd10, loinc).
        
        Args:
            mapping_name: Name of mapping ('mesh', 'snomed', 'atc', 'icd10', 'loinc')
            
        Returns:
            Mapping dictionary
        """
        if mapping_name not in self._mapping_files:
            mapping_paths = {
                'mesh': self.config.MESH_MAP_FILE,
                'snomed': self.config.SNOMED_MAP_FILE,
                'atc': self.config.ATC_MAP_FILE,
                'icd10': self.config.ICD10_MAP_FILE,
                'loinc': self.config.LOINC_MAP_FILE,
            }
            
            if mapping_name not in mapping_paths:
                raise ValueError(f"Unknown mapping: {mapping_name}")
            
            self._mapping_files[mapping_name] = read_json_file(mapping_paths[mapping_name])
        
        return self._mapping_files[mapping_name]
