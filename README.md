# Knowledge-Enhanced LLMs for Multilingual Biomedical Concept Normalization — A Multilingual Benchmarking and Behavioral Analysis

A modular pipeline for **medical concept normalization** — mapping clinical and biomedical terms to standardized UMLS (Unified Medical Language System) concepts. The system leverages dense retrieval and LLM-based reranking across 10 datasets in 5 languages.

## Pipeline Overview

<p align="center">
<img width="2236" height="970" alt="overview" src="https://github.com/user-attachments/assets/fe8729b7-f740-4423-b0a4-8caead54ee0a" />
</p>


## Modules

| Module | Description |
|--------|-------------|
| [`umls_preprocessing`](https://github.com/hrouhizadeh/multilingual_concept_normalization/tree/main/src/bm25) | Extract definitions, semantic types, hierarchies, and preferred terms from UMLS RRF files |
| [`dataset_preprocessing`](./dataset_preprocessing/) | Convert 10 biomedical NER/normalization datasets into a unified CSV format |
| [`bm25_retrieval`](./bm25_retrieval/) | BM25-based candidate retrieval using Elasticsearch |
| [`generating_embeddings`](./generating_embeddings/) | Generate UMLS term embeddings with generative (Qwen) and discriminative (E5, BGE) models |
| [`llm_reranker`](./llm_reranker/) | LLM-based reranking of candidate concepts using UMLS features and chain-of-thought reasoning |

## Supported Datasets

| Dataset | Language | Ontology |
|---------|----------|----------|
| BC5CDR | English | MeSH |
| N2C2 | English | RxNorm, SNOMED |
| Quaero | French | UMLS |
| MedUCD | French | ATC |
| DisTEMIST | Spanish | SNOMED CT |
| PharmaCoNER | Spanish | SNOMED CT |
| BRONCO | German | ATC, ICD-10 |
| TLLV | Turkish | LOINC |
| MANTRA-GSC | Multilingual | MedDRA, MeSH, SNOMED |
| XL-BEL | Multilingual | UMLS |

## Getting Started

### Prerequisites

- Python 3.8+
- [UMLS License](https://uts.nlm.nih.gov/uts/) (for UMLS data access)
- Elasticsearch (for BM25 retrieval)
- GPU(s) (for embedding generation and LLM reranking)

### Installation

```bash
pip install -r requirements.txt
```

### Step-by-Step Usage

#### Step 1 — Preprocess UMLS

Extract structured JSON files from raw UMLS RRF data:

```bash
cd umls_preprocessing
python run_all.py
```

This produces: `umls_definitions.json`, `cui_semantic_mapping.json`, `umls_hierarchies.json`, `umls_preferred_terms.json`, and `umls_all_terms.jsonl`.

#### Step 2 — Preprocess Datasets

Convert raw datasets into a unified CSV format:

```bash
cd dataset_preprocessing
python run_all.py
```

Outputs unified CSVs with columns: `term`, `code`, `language`, `semantic_group`, `exact_match`, `source`.

#### Step 3a — BM25 Retrieval

Index UMLS terms in Elasticsearch, then retrieve candidates:

```bash
cd bm25_retrieval
python query_bm25.py
python eval.py
```

#### Step 3b — Dense Retrieval

Generate embeddings for UMLS terms and query terms:

```bash
cd generating_embeddings/generative
python embedder.py

# or discriminative models
cd generating_embeddings/discriminative
python embedder.py
```

#### Step 4 — LLM Reranking

Rerank candidates using an LLM with UMLS feature ablation:

```bash
cd llm_reranker
python umls_candidate_reranker.py
```

Runs a full experiment grid over top-k values × feature combinations × retrieval models.

## Repository Structure

```
/
├── README.md                          # This file
├── requirements.txt
│
├── umls_preprocessing/                # Step 1: UMLS data extraction
│   ├── config.py
│   ├── run_all.py
│   ├── scripts/
│   │   ├── extract_definitions.py
│   │   ├── extract_semantic_types.py
│   │   ├── extract_hierarchies.py
│   │   ├── extract_preferred_terms.py
│   │   └── extract_all_terms.py
│   ├── data/umls/META/                # Place UMLS RRF files here
│   └── output/
│
├── dataset_preprocessing/             # Step 2: Dataset conversion
│   ├── config.py
│   ├── utils.py
│   ├── run_all.py
│   ├── scripts/
│   │   ├── preprocess_bc5cdr.py
│   │   ├── preprocess_n2c2.py
│   │   └── ...
│   ├── data/
│   │   ├── datasets/                  # Raw dataset files
│   │   └── mappings/                  # UMLS mapping files
│   └── output/
│
├── bm25_retrieval/                    # Step 3a: Lexical retrieval
│   ├── config.py
│   ├── query_bm25.py
│   └── eval.py
│
├── generating_embeddings/             # Step 3b: Dense retrieval
│   ├── generative/
│   │   ├── config.py
│   │   ├── utils.py
│   │   └── embedder.py
│   └── discriminative/
│       ├── config.py
│       ├── utils.py
│       └── embedder.py
│
└── llm_reranker/                      # Step 4: LLM reranking
    ├── config.py
    └── umls_candidate_reranker.py
```

## Configuration

All modules use **project-relative paths** by default — no hardcoded user paths. Each module's `config.py` can be customized via:

- Direct editing of the config file
- Environment variables (see each module's README for details)
