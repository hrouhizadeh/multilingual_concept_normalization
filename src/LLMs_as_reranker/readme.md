# LLMs as reranker

Knowledge-enhanced candidate reranking for biomedical concept normalization using generative LLMs. This module takes dense retrieval candidates and reranks them by enriching each candidate with structured UMLS knowledge and prompting a generative LLM to select the best match.

## Overview

The reranker operates as **Stage 4** of the normalization pipeline:

1. Reads CSV files produced by the dense retrieval stage (Stage 3)
2. Enriches each candidate CUI with UMLS knowledge (definitions, synonyms, hierarchies, semantic groups)
3. Constructs structured prompts including the query's target ontology and semantic group
4. Runs batch inference via [vLLM](https://github.com/vllm-project/vllm) and saves ranked predictions

## Prerequisites

### Hardware

- GPU with sufficient VRAM for the target model (e.g., ~80 GB for Qwen3-32B)

### Software

```bash
pip install -r requirements.txt
```

Required packages:
- `vllm` — high-throughput LLM inference
- `pandas` — data processing
- `torch` — PyTorch backend

### UMLS knowledge files

The reranker requires pre-processed UMLS JSON files (produced by the `umls_preprocessing` module):

| File | Description |
|------|-------------|
| `umls_preferred_terms.json` | Preferred English terms per CUI |
| `cui_synonyms.json` | Synonyms grouped by language |
| `umls_definitions.json` | Definitions grouped by source vocabulary |
| `umls_hierarchies.json` | Hypernym/hyponym relationships |
| `cui_semantic_mapping.json` | Semantic types and groups |

### Input format

CSV files from the dense retrieval stage with at minimum these columns:

| Column | Description |
|--------|-------------|
| `term` | Input biomedical term |
| `code` | Gold-standard CUI |
| `retrieved_codes` | List of candidate CUIs (as string) |
| `retrieved_terms` | List of candidate term strings (as string) |
| `retrieved_scores` | List of retrieval similarity scores (as string) |
| `exact_match` | Whether the term was an exact match (0/1) |
| `target_ontologies` | Target ontology for normalization |
| `semantic_group` | Semantic group of the query term |

## Usage

### Basic usage

```bash
python rerank.py \
    --input-dir outputs/dense_retrieval \
    --output-dir outputs/reranker \
    --umls-dir data/umls
```

### Custom model and top-K

```bash
python rerank.py \
    --model Qwen/Qwen3-32B \
    --top-k 10 25 50 \
    --umls-dir data/umls
```

### Feature ablation experiments

Run specific UMLS knowledge combinations:

```bash
# Single combination
python rerank.py --features "synonyms,hierarchy"

# Multiple combinations (grid search)
python rerank.py \
    --features \
        "No_umls_knowledge" \
        "definition" \
        "synonyms" \
        "definition,synonyms" \
        "hierarchy" \
        "synonyms,hierarchy" \
        "definition,hierarchy" \
        "definition,synonyms,hierarchy"
```

Available features: `preferred_term`, `synonyms`, `semantic_groups`, `definition`, `hierarchy`.

### Knowledge feature limits

Control how many synonyms and hierarchy terms are included per candidate:

```bash
python rerank.py --max-synonyms 10 --max-hierarchy 8
```

### Process specific models and splits

```bash
python rerank.py \
    --dr-llms bge-m3 e5-mistral \
    --splits test dev
```

### Disable thinking mode

```bash
python rerank.py --no-thinking
```

### All options

```bash
python rerank.py --help
```

## Configuration

Default values are defined in `config.py`. CLI arguments override config values. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-32B` | HuggingFace model identifier |
| `ENABLE_THINKING` | `True` | Enable chain-of-thought reasoning |
| `BATCH_SIZE` | `100` | Inference batch size |
| `TOP_CANDIDATES_LIST` | `[10, 25, 50]` | Top-K values for reranking |
| `MAX_SYNONYMS` | `5` | Max synonyms per candidate |
| `MAX_HIERARCHY` | `5` | Max hypernyms and hyponyms each per candidate |
| `FEATURES_FOR_EXP` | See config | Feature combinations for ablation |

## Output

Results are saved with the following directory structure:

```
outputs/reranker/
└── {top_k}/
    └── {model_name}/
        └── {dr_llm}/
            └── {features}/
                └── {split}/
                    └── {dataset}_results_{timestamp}.csv
```

Each output CSV contains all original columns plus:

| Column | Description |
|--------|-------------|
| `prediction` | Predicted candidate index (1-based) |
| `thinking_content` | Raw LLM output including chain-of-thought |
| `predicted_cui` | Resolved CUI from the prediction |

## Project structure

```
llm_reranker/
├── README.md          # This file
├── config.py          # Default configuration
├── rerank.py          # Main reranking script
└── requirements.txt   # Python dependencies
```
