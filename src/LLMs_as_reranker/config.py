"""Configuration for UMLS candidate reranking with generative LLMs.

This module defines all configurable parameters for the reranking pipeline,
including model settings, file paths, and feature flags for UMLS knowledge
infusion experiments.
"""


# =============================================================================
# Model settings
# =============================================================================
MODEL_NAME = "Qwen/Qwen3-32B"
ENABLE_THINKING = True
MAX_TOKENS = 4096
TEMPERATURE = 0.7
MAX_MODEL_LEN = 32000
TENSOR_PARALLEL_SIZE = 2
GPU_MEMORY_UTILIZATION = 0.90
TOP_P = 0.95
MIN_P = 0.0
SEED = 42
BATCH_SIZE = 100

# =============================================================================
# Data paths (override via CLI arguments)
# =============================================================================
INPUT_DIR = "outputs/dense_retrieval"
OUTPUT_DIR = "outputs/reranker"

# UMLS knowledge files
SYNONYM_FILE = "data/umls/cui_synonyms.json"
PREFERRED_TERM_FILE = "data/umls/umls_preferred_terms.json"
HIERARCHY_FILE = "data/umls/umls_hierarchies.json"
DEFINITION_FILE = "data/umls/umls_definitions.json"
SEMANTIC_FILE = "data/umls/cui_semantic_mapping.json"

# =============================================================================
# Knowledge feature limits
# =============================================================================
MAX_SYNONYMS = 5
MAX_HIERARCHY = 5  # max hypernyms and hyponyms each

# =============================================================================
# Experiment settings
# =============================================================================

# Dense retrieval models to process ('all' to auto-detect from INPUT_DIR)
DR_LLMS = ["nemotron"]

# Dataset splits to process
SPLITS = ["test"]

# Number of top candidates to rerank (list for grid search)
TOP_CANDIDATES_LIST = [10, 25, 50]

# Feature combinations to evaluate (list of lists for ablation experiments)
# Available features: preferred_term, synonyms, semantic_groups,
#                     definition, hierarchy
FEATURES_FOR_EXP = [
    ["No_umls_knowledge"],
    ["definition"],
    ["synonyms"],
    ["definition", "synonyms"],
    ["hierarchy"],
    ["synonyms", "hierarchy"],
    ["definition", "hierarchy"],
    ["definition", "synonyms", "hierarchy"],
]