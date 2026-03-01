# ============================================================
# Configuration for UMLS Embedding Pipeline
# ============================================================
# This config is shared across all pipeline steps:
#   1. generate_embeddings.py  - Generate embeddings from UMLS terms
#   2. index_qdrant.py         - Index embeddings into Qdrant
#   3. query_qdrant.py         - Query Qdrant for candidate retrieval
# ============================================================

# --- Model ---
# Llama-Embed-Nemotron-8B: fine-tuned from Llama-3.1-8B with bidirectional
# attention. 32 hidden layers, embedding dim = 1024.
# Reference: https://huggingface.co/nvidia/llama-embed-nemotron-8b
model_name = "nvidia/llama-embed-nemotron-8b"
attention_implementation = "flash_attention_2"
torch_dtype = "bfloat16"
padding_side = "left"
trust_remote_code = True

# Derived model save name (used in file paths and collection names)
model_save_name = model_name.replace("/", "-").replace("_", "-")

# --- Step 1: Embedding Generation ---
embedding_input_file = "data/umls2025_with_final_train_data_2csv.csv"
embedding_output_dir = "embeddings"
embedding_batch_size = 500
embedding_save_every = 10000
embedding_exclude_columns = ["term", "CUI", "ENG", "FRE", "GER", "SPA", "TUR"]

# --- Step 2: Qdrant Indexing ---
qdrant_url = "http://localhost:6340"
qdrant_collection_name = "nvidiallama"
qdrant_upload_batch_size = 200
qdrant_distance = "Cosine"
qdrant_quantization_type = "int8"
qdrant_quantization_quantile = 0.99
qdrant_memmap_threshold = 20000

# --- Step 3: Query / Retrieval ---
query_data_dir = "data/all"
query_output_dir = "outputs/dr_output"
query_splits = ["dev", "test"]
query_top_k_fetch = 500       # Number of candidates fetched from Qdrant
query_top_k_save = 100        # Number of candidates saved to output
query_batch_size = 50         # Batch size for query encoding
cui_semantic_mapping_path = "data/umls_mappings/cui_semantic_mapping.json"

# Dataset-to-ontology mapping for filtering Qdrant results
main_ontology_list = [
    "ATC", "CPT", "CPTSP", "DMDICD10", "DMDUMD",
    "DRUGBANK", "ICD10", "ICD10AE", "ICD10AM", "ICD10AMAE", "ICD10CM", "ICD10PCS",
    "ICD9CM", "ICPC", "ICPC2EENG", "ICPC2ICD10ENG", "ICPC2P", "ICPCFRE", "ICPCGER",
    "ICPCSPA", "LNC", "LNC-DE-AT", "LNC-DE-DE", "LNC-ES-AR", "LNC-ES-ES", "LNC-ES-MX",
    "LNC-FR-BE", "LNC-FR-CA", "LNC-FR-FR", "LNC-TR-TR", "MDR", "MDRFRE", "MDRGER",
    "MDRSPA", "MED-RT", "MEDCIN", "MEDLINEPLUS", "MEDLINEPLUS_SPA", "MSH", "MSHFRE",
    "MSHGER", "MSHSPA", "MTH", "MTHCMSFRF", "MTHICD9", "MTHICPC2EAE", "MTHICPC2ICD10AE",
    "MTHMST", "MTHMSTFRE", "MTHSPL", "RXNORM", "NCBI", "SCTSPA", "SNOMEDCT_US",
    "SNOMEDCT_VET", "WHO", "WHOFRE", "WHOGER", "WHOSPA",
]

source_ontology_dict = {
    "bc5cdr":    ["MSH", "BC5CDR"],
    "n2c2":      ["SNOMEDCT_US", "SNOMEDCT_VET", "RXNORM", "N2C2"],
    "bronco":    ["ICD10", "ATC", "BRONCO"],
    "distemist": ["SNOMEDCT_US", "SNOMEDCT_VET", "SCTSPA", "DISTEMIST"],
    "pharma":    ["SNOMEDCT_US", "SNOMEDCT_VET", "SCTSPA", "PHARMA"],
    "quaero":    main_ontology_list + ["QUAERO"],
    "tllv":      ["LNC", "LNC-TR-TR", "TLLV"],
    "ucd":       ["ATC", "UCD"],
    "xl-bel":    main_ontology_list,
    "gsc":       [
        "SNOMEDCT_US", "SNOMEDCT_VET", "SCTSPA",
        "MDR", "MDRFRE", "MDRGER", "MDRSPA",
        "MSH", "MSHFRE", "MSHGER", "MSHSPA",
    ],
}
