# Knowledge-Enhanced LLMs for Multilingual Biomedical Concept Normalization — A Multilingual Benchmarking and Behavioral Analysis

A modular pipeline for biomedical concept normalization — mapping clinical and biomedical terms to standardized UMLS (Unified Medical Language System) concepts. The system combines dense retrieval with knowledge-enhanced LLM reranking, evaluated on MedLexAlign, a unified multilingual benchmark built from 10 datasets across 5 languages (English, French, German, Spanish, and Turkish).

## Pipeline overview

<p align="center">
<img width="1860" height="769" alt="OV" src="https://github.com/user-attachments/assets/cc7615ea-f448-45dd-980e-2e89f7030aa5" />
</p>


## Modules
| Module | Description |
|--------|-------------|
| [`umls_preprocess`](https://github.com/hrouhizadeh/multilingual_concept_normalization/tree/main/src/umls_preprocess) | Extract definitions, synonyms, semantic groups, hierarchies, and preferred terms from UMLS RRF files |
| [`dataset_preprocessing`](https://github.com/hrouhizadeh/multilingual_concept_normalization/tree/main/src/dataset_prepocessing) | Convert 10 biomedical concept normalization datasets into a unified format |
| [`bm25`](https://github.com/hrouhizadeh/multilingual_concept_normalization/tree/main/src/bm25) | BM25-based candidate retrieval using Elasticsearch |
| [`embedding_generation`](https://github.com/hrouhizadeh/multilingual_concept_normalization/tree/main/src/embedding_generation) | Generate UMLS term embeddings with generative and discriminative large language models |
| [`LLMs_as_rerankers`](https://github.com/hrouhizadeh/multilingual_concept_normalization/tree/main/src/LLMs_as_rerankers) | LLM-based reranking of candidate concepts using UMLS features and chain-of-thought reasoning |


## Supported datasets

| Dataset | Language | Ontology |
|---------|----------|----------|
| BC5CDR | English | MeSH |
| N2C2 | English | RxNorm, SNOMED CT|
| Quaero | French | UMLS |
| Med-UCD | French | ATC |
| DisTEMIST | Spanish | SNOMED CT |
| PharmaCoNER | Spanish | SNOMED CT |
| BRONCO | German | ATC, ICD-10 |
| TLLV | Turkish | LOINC |
| MANTRA-GSC | Multilingual | MedDRA, MeSH, SNOMED CT |
| XL-BEL | Multilingual | UMLS |

## Getting started

### Prerequisites

- Python 3.8+
- [UMLS License](https://uts.nlm.nih.gov/uts/) (for UMLS data access)
- Elasticsearch (for BM25 retrieval)
- GPU(s) (for embedding generation and LLM reranking)

### Installation

```bash
pip install -r requirements.txt
```

### Step-by-Step usage

#### Step 1 — Preprocess UMLS

Extract structured JSON files from raw UMLS RRF data:

```bash
cd umls_preprocessing
python run_all.py
```

This produces: `umls_definitions.json`, `cui_semantic_mapping.json`, `umls_hierarchies.json`, `umls_preferred_terms.json`, and `umls_all_terms.jsonl`.

#### Step 2 — Preprocess datasets

Convert raw datasets into a unified CSV format:

```bash
cd dataset_preprocessing
python run_all.py
```

Outputs unified CSVs with columns: `term`, `code`, `language`, `semantic_group`, `source`.

#### Step 3a — BM25 retrieval

Index UMLS terms in Elasticsearch, then retrieve candidates:

```bash
cd bm25_retrieval
python query_bm25.py
python eval.py
```

#### Step 3b — Dense retrieval

Generate embeddings for UMLS terms and query terms:

```bash
cd generating_embeddings/generative
python embedder.py

# or discriminative models
cd generating_embeddings/discriminative
python embedder.py
```
Store the embeddings in a Qdrant vector database and retrieve the terms most similar to the input query.

#### Step 4 — LLM reranking

Rerank candidates using an LLM with the UMLS kowledge:

```bash
cd llm_reranker
python umls_candidate_reranker.py
```

Runs a full experiment grid over top-k values × feature combinations × retrieval models.

## Repository structure

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

