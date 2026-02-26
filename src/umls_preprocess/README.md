# UMLS preprocessing

A collection of Python scripts to extract and preprocess data from UMLS (Unified Medical Language System) RRF files into JSON format for downstream NLP tasks.

## Overview

This module processes UMLS RRF (Rich Release Format) files and generates JSON outputs containing:

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `extract_definitions.py` | MRDEF.RRF | `umls_definitions.json` | Concept definitions by source |
| `extract_semantic_types.py` | MRSTY.RRF, SemGroups.txt | `cui_semantic_mapping.json` | Semantic types and groups |
| `extract_hierarchies.py` | MRREL.RRF | `umls_hierarchies.json` | Hypernym/hyponym relationships |
| `extract_preferred_terms.py` | MRCONSO.RRF | `umls_preferred_terms.json` | Preferred English terms |
| `extract_all_terms.py` | MRCONSO.RRF | `umls_all_terms.jsonl` | All terms by source/language |

## Prerequisites

### 1. Download UMLS

You need a UMLS license to download the data:

1. **Create a UMLS account**: Go to [UMLS Terminology Services](https://uts.nlm.nih.gov/uts/) and sign up
2. **Request a license**: Complete the license agreement
3. **Download UMLS**: Go to [UMLS Knowledge Sources](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html)
4. **Extract the files**: After downloading, extract the `META` folder containing the `.RRF` files

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `pandas` (for data processing)
- `jsonlines` (for JSONL output)

## Setup

### 1. Clone/copy this folder to your project

```bash
# If part of a larger project
cp -r umls_preprocessing /path/to/your/project/
```

### 2. Place UMLS files in the data directory

Copy the required RRF files to `data/umls/META/`:

```
umls_preprocessing/
└── data/
    └── umls/
        └── META/
            ├── MRCONSO.RRF    # Required for terms extraction
            ├── MRDEF.RRF      # Required for definitions
            ├── MRREL.RRF      # Required for hierarchies
            ├── MRSTY.RRF      # Required for semantic types
            └── SemGroups.txt  # Required for semantic groups
```

## Usage

### Run all scripts

```bash
# Run all preprocessing scripts in sequence
python run_all.py

# Check files first without running
python run_all.py --check

# Skip memory-intensive all-terms extraction
python run_all.py --skip-all-terms
```

### Run individual scripts

Each script can be run independently:

```bash
# Extract definitions
python scripts/extract_definitions.py

# Extract semantic types
python scripts/extract_semantic_types.py

# Extract hierarchies
python scripts/extract_hierarchies.py

# Extract preferred terms
python scripts/extract_preferred_terms.py

# Extract all terms 
python scripts/extract_all_terms.py
```

### Custom Input/Output Paths

All scripts support custom paths via command-line arguments:

```bash
# Custom paths
python scripts/extract_definitions.py \
    --input /path/to/MRDEF.RRF \
    --output /path/to/output.json

# Extract Spanish preferred terms
python scripts/extract_preferred_terms.py --language SPA

# See all options
python scripts/extract_definitions.py --help
```

## Output Formats

### umls_definitions.json

Definitions grouped by CUI and source vocabulary:

```json
[
  {
    "C0000039": {
      "MSH": "A phospholipid...",
      "NCI": "A synthetic phospholipid..."
    }
  }
]
```

### cui_semantic_mapping.json

Semantic type and group mappings:

```json
{
  "C0000039": {
    "sem_T_code": ["T109", "T121"],
    "sem_type": ["Organic Chemical", "Pharmacologic Substance"],
    "sem_group": ["Chemicals & Drugs", "Chemicals & Drugs"]
  }
}
```

### umls_hierarchies.json

Parent/child (hypernym/hyponym) relationships:

```json
{
  "C0000039": {
    "hypernyms": ["C0031676", "C0001128"],
    "hyponyms": ["C0123456"]
  }
}
```

### umls_preferred_terms.json

Preferred English terms for each CUI:

```json
{
  "C0000039": "1,2-dipalmitoylphosphatidylcholine",
  "C0000052": "1,4-alpha-Glucan Branching Enzyme"
}
```

### umls_all_terms.jsonl

Comprehensive term data (JSON Lines format):

```json
{"CUI": "C0000039", "MSH": [{"CODE": "D015060", "STR": "...", "LAT": "ENG"}], "terms_ENG": "...", "terms_all": "..."}
```

## Configuration

Edit `config.py` to customize paths:

```python
# Point to your UMLS installation
UMLS_META_DIR = Path("/your/custom/path/to/META")

# Custom output directory
OUTPUT_DIR = Path("/your/custom/output/path")
```

```bash
python run_all.py --skip-all-terms
```

## Project structure

```
umls_preprocessing/
├── README.md                      # This file
├── config.py                      # Centralized path configuration
├── requirements.txt               # Python dependencies
├── run_all.py                     # Run all scripts in sequence
├── data/
│   └── umls/
│       └── META/                  # Place UMLS RRF files here
│           ├── MRCONSO.RRF
│           ├── MRDEF.RRF
│           ├── MRREL.RRF
│           ├── MRSTY.RRF
│           └── SemGroups.txt
├── output/                        # Generated JSON files
│   ├── umls_definitions.json
│   ├── cui_semantic_mapping.json
│   ├── umls_hierarchies.json
│   ├── umls_preferred_terms.json
│   └── umls_all_terms.jsonl
└── scripts/
    ├── __init__.py
    ├── extract_definitions.py
    ├── extract_semantic_types.py
    ├── extract_hierarchies.py
    ├── extract_preferred_terms.py
    └── extract_all_terms.py
```

## License

This preprocessing code is provided as-is. Note that UMLS data requires a separate license from the National Library of Medicine.

## References

- [UMLS Documentation](https://www.nlm.nih.gov/research/umls/implementation_resources/documentation.html)
- [UMLS File Formats](https://www.ncbi.nlm.nih.gov/books/NBK9685/)
- [Semantic Types](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/SemanticTypesAndGroups.html)
