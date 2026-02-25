# Dataset preprocessing

Preprocessing scripts for 10 concept normalization datasets across 5 languages. Converts various dataset formats into a unified CSV format.

## Supported datasets

| Dataset | Language | Target Ontology | Source |
|---------|----------|-----------------|--------|
| BC5CDR | English | MeSH | [BioCreative V](https://github.com/JHnlp/BioCreative-V-CDR-Corpus) |
| N2C2 | English | RxNorm, SNOMED | [n2c2 2019 challenge](https://n2c2.dbmi.hms.harvard.edu/2019-track-3) |
| Quaero | French | UMLS | [QUAERO corpus](https://quaerofrenchmed.limsi.fr/) |
| MedUCD | French | ATC | [atih](https://www.atih.sante.fr/unites-communes-de-dispensation-prises-en-charge-en-sus) |
| DisTEMIST | Spanish | SNOMED CT | [DisTEMIST](https://temu.bsc.es/distemist/) |
| Pharma | Spanish | SNOMED CT | [PharmaCoNER](https://temu.bsc.es/pharmaconer/)|
| BRONCO | German | ATC, ICD-10 | [BRONCO150](https://www2.informatik.hu-berlin.de/~leser/bronco/index.html) |
| TLLV | Turkish | LOINC | Turkish lab test mappings |
| MANTRA-GSC | Multilingual | MedDRA, MeSH, SNOMED | [A multilingual gold-standard corpus for biomedical concept recognition](https://pmc.ncbi.nlm.nih.gov/articles/PMC4986661/) |
| XL-BEL | Multilingual | UMLS | [XL-BEL](https://github.com/cambridgeltl/sapbert/tree/main/evaluation) |

## Prerequisites

### 1. Process UMLS data first

This module depends on outputs from the `umls_preprocessing` module. Make sure you have:

1. Run the UMLS preprocessing scripts
2. Generated the required JSON files:
   - `cui_semantic_mapping.json` - Semantic type mappings
   - `word_mapping_onto.json` - Term to ontology mappings  
   - `cui_codes.json` - List of valid CUI codes
   - Vocabulary mapping files (`mesh_map.json`, `snomed_map.json`, etc.)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `pandas` - Data manipulation
- `scikit-learn` - Train/test splitting
- `openpyxl` - Excel file reading
- `xlrd` - Legacy Excel file reading
- `numpy` - Numerical operations

### 3. Download datasets

Download each dataset from its source and place it in the appropriate directory.

Note that n2c2 and bronco150 datasets are not publicly available. You must follow the instructions by the respective data providers to request access: for n2c2, register at https://n2c2.dbmi.hms.harvard.edu/ and sign a DUA; for BRONCO150, contact the dataset creators to obtain permission (https://www2.informatik.hu-berlin.de/~leser/bronco/index.html).



## Setup

### 1. Directory structure

```
dataset_preprocessing/
├── config.py                 # Path configuration
├── utils.py                  # Shared utilities
├── requirements.txt
├── run_all.py               # Run all preprocessing
├── data/
│   ├── datasets/            # Raw dataset files
│   │   ├── BC5CDR/
│   │   │   └── source_files/
│   │   │       ├── CDR_TrainingSet.PubTator.txt
│   │   │       ├── CDR_DevelopmentSet.PubTator.txt
│   │   │       └── CDR_TestSet.PubTator.txt
│   │   ├── N2C2/
│   │   │   └── source_files/
│   │   │       ├── train_norm/
│   │   │       ├── train_note/
│   │   │       ├── test_norm/
│   │   │       └── test_note/
│   │   ├── Quaero/
│   │   │   └── source_files/
│   │   │       ├── train/
│   │   │       ├── dev/
│   │   │       └── test/
│   │   └── ... (other datasets)
│   └── mappings/            # UMLS mapping files
│       ├── word_mapping_onto.json
│       ├── cui_codes.json
│       ├── mesh_map.json
│       ├── snomed_map.json
│       ├── atc_map.json
│       ├── icd10_map.json
│       └── loinc_map.json
├── output/
│   ├── all/                 # Combined outputs
│   │   ├── train/
│   │   ├── dev/
│   │   └── test/
│   └── by_dataset/          # Per-dataset outputs
└── scripts/
    ├── preprocess_bc5cdr.py
    ├── preprocess_n2c2.py
    └── ... (other scripts)
```

### 2. Configure paths

Edit `config.py` to match your setup:

```python
# Point to your UMLS preprocessing output
UMLS_OUTPUT_DIR = Path("/path/to/umls_preprocessing/output")

# Point to your datasets
DATASETS_BASE_DIR = Path("/path/to/your/datasets")
```

### 3. Verify Setup

```bash
python config.py  # Shows configuration and file status
python run_all.py --check  # Validates all inputs
```

## Usage

### Run all datasets

```bash
# Process all available datasets
python run_all.py

# Check files without processing
python run_all.py --check

# List available datasets
python run_all.py --list
```

### Run specific datasets

```bash
# Process only specific datasets
python run_all.py --only bc5cdr n2c2 quaero

# Skip certain datasets
python run_all.py --skip turkish bronco
```

### Run individual scripts

```bash
# Each script can be run independently
python scripts/preprocess_bc5cdr.py
python scripts/preprocess_n2c2.py --dataset-dir /custom/path

# Get help for any script
python scripts/preprocess_bc5cdr.py --help
```

## Output format

All scripts produce CSV files with a unified schema:

| Column | Description |
|--------|-------------|
| `term` | The medical term/mention |
| `code` | UMLS CUI code(s) |
| `langauge` | Language code (eng, fre, spa, ger, tur) |
| `semantic_type` | UMLS semantic type(s) |
| `semantic_group` | UMLS semantic group(s) |
| `targe_ontologies` | Source ontologies |
| `exact_match` | Exact match status (0-3) |
| `source` | Dataset source identifier |

### Exact match status codes

| Code | Meaning |
|------|---------|
| 0 | Not in train/dev, not in target ontology |
| 1 | In target ontology only |
| 2 | In train/dev set only |
| 3 | In both train/dev and target ontology |

## Dataset-specific notes

### BC5CDR
- PubTator format with MeSH annotations
- Requires `mesh_map.json` for code mapping

### N2C2
- Paired .norm and .txt files
- Clinical notes with character offset annotations
- Automatically split into train/dev (80/20)
- Requires `rxnorm_map.json` and `snomed_map.json` for code mapping

### Quaero
- BRAT annotation format (.ann files)
- French medical corpus (MEDLINE + EMEA)

### MedUCD
- Excel files (codes.xlsx, labels.xlsx)
- French medication terms with ATC codes
- Split into train/dev/test (60/20/20)
- Requires `atc_map.json` for code mapping

### DisTEMIST
- TSV format with SNOMED codes
- Only EXACT semantic relations used
- Automatically split into train/dev
- Requires `snomed_map.json` for code mapping

### Pharma
- BRAT format with SNOMED annotations
- Has separate train/valid/test folders
- Requires `snomed_map.json` for code mapping

### BRONCO
- German clinical texts
- Mixed ATC and ICD-10 codes
- Limited files, split by document
- Requires `atc_map.json` and `icd10_map.json` for code mapping

### Turkish LOINC
- Excel file with lab test mappings
- Multiple columns concatenated for term
- Split into train/dev/test (60/20/20)
- Requires `loinc_map.json` for code mapping

### MANTRA-GSC
- XML format with CUI annotations
- Test-only dataset (no train/dev)
- Multiligual: EN, DE, ES, FR

### XL-BEL
- Simple text format (CUI||term)
- Test-only dataset
- Multiligual: EN, DE, ES, TR
