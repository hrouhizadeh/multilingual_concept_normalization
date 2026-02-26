# BM25 retrieval pipeline

## Overview

This module performs **BM25-based candidate retrieval** from an Elasticsearch index for medical/clinical term normalization tasks. Given input terms (e.g., from EHR data), it queries an Elasticsearch index and returns ranked candidate CUIs (Concept Unique Identifiers) with their BM25 scores.

## Files

| File | Description |
|------|-------------|
| `config.py` | Central configuration (ES connection, paths, search params, filtering) |
| `query_bm25.py` | Main retrieval script — reads CSVs, queries Elasticsearch, saves results |
| `eval.py` | Evaluation script — computes Recall@k metrics over output files |

## Prerequisites

- **Python 3.8+**
- **Elasticsearch** running and accessible (default: `http://localhost:9400`)
- An indexed corpus with document IDs corresponding to CUIs

### Python dependencies

```bash
pip install elasticsearch pandas
```

## Configuration

All settings are centralized in `config.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ELASTIC_URL` | Elasticsearch endpoint | `http://localhost:9400` |
| `INDEX_NAME` | Name of the ES index to query | `umls_concepts` |
| `TOP_K` | Max candidates to retrieve per query | `1000` |
| `INPUT_FIELD` | ES field to match against | `term_all` |
| `APPLY_EXACT_MATCH_FILTER` | Whether to filter input rows by `exact_match` column | `True` |
| `FILTER_VALUE` | Value to filter on (e.g., `0` = no exact match found) | `0` |
| `INPUT_PATH` | Directory containing input CSV files | `<project>/data/dev` |
| `OUTPUT_BASE` | Base directory for outputs | `<project>/outputs` |

All paths are **relative to the project root** by default, so no manual editing is needed. You can also override them via environment variables:

```bash
export BM25_INPUT_PATH=/your/custom/input/path
export BM25_OUTPUT_BASE=/your/custom/output/path
```

## Input format

Input CSV files must contain at a minimum:

- `term` — the query text to search
- `code` — ground-truth CUI(s) (as a list) for evaluation
- `exact_match` *(optional)* — used for filtering if `APPLY_EXACT_MATCH_FILTER` is enabled

## Usage

### 1. Run BM25 retrieval

```bash
python query_bm25.py
```

This will:
1. Read all `.csv` files from `INPUT_PATH`
2. Optionally filter rows based on the `exact_match` column
3. Query Elasticsearch for each term using BM25
4. Save results with `retrieved_cuis` and `retrieved_cuis` columns to `SAVE_PATH`

### 2. Evaluate results

Update the `folder` variable in `eval.py` to point to your output directory, then run:

```bash
python eval.py
```

This computes **Recall@k** for k ∈ {1, 3, 5, 10} across all output CSV files and saves a summary to `recall_summary_with_counts.csv`.

## Output format

Each output CSV contains the original columns plus:

- `candidate_cuis` — list of retrieved CUI IDs ranked by BM25 score
- `candidate_scores` — corresponding BM25 scores
