# Generating embeddings

A unified framework for generating UMLS medical term embeddings using both generative (Qwen-Embedding via vLLM) and discriminative (SentenceTransformers) language models.

## Project structure

```
generating_embeddings/
├── README.md
├── requirements.txt
│
├── generative/                    # Qwen-Embedding models (vLLM)
│   ├── config.py                  # Configuration for generative models
│   ├── utils.py                   # Shared utility functions
│   └── embedder.py                # QwenEmbedder class & pipeline
│
└── discriminative/                # SentenceTransformer models
    ├── config.py                  # Configuration for discriminative models
    ├── utils.py                   # Shared utility functions
    └── embedder.py                # DiscriminativeEmbedder class & pipeline
```

## Installation

### Base dependencies

```bash
pip install pandas numpy
```

### For generative models (Qwen-Embedding)

```bash
pip install vllm torch
```

### For discriminative models (SentenceTransformers)

```bash
pip install sentence-transformers torch
```

### Full installation

```bash
pip install -r requirements.txt
```

## Quick start

### Using Generative Models (Qwen-Embedding)

```bash
cd generating_embeddings/generative
python embedder.py
```

### Using discriminative models (E5, BERT, RoBERTa, etc.)

```bash
cd generating_embeddings/discriminative
python embedder.py
```

## Configuration

Both modules use a `config.py` file with a dataclass for easy configuration.

### Generative Model Configuration

```python
from config import EmbeddingConfig

config = EmbeddingConfig(
    # Model settings
    model_name="Qwen/Qwen3-Embedding-8B",  # or 4B, 0.6B
    embedding_dim=1024,                     # Matryoshka dimensions: 128, 256, 512, 1024, etc.
    max_length=8192,
    instruction="Given a web search query, retrieve relevant passages that answer the query",
    
    # Processing
    batch_size=100,
    save_every=10000,
    
    # Paths
    input_file="/path/to/your/data.csv",
    output_dir="embeddings/custom-output",
    
    # Columns to skip
    exclude_columns=["word", "CUI", "ENG", "FRE", "GER", "SPA", "TUR"],
)
```

### Discriminative model configuration

```python
from config import EmbeddingConfig, MODELS

config = EmbeddingConfig(
    # Model settings
    model_name=MODELS["e5-large"],  # or any HuggingFace model
    device="cuda:0",                # None for auto-detect
    
    # Processing
    batch_size=50,
    show_progress_bar=False,
    
    # Paths
    input_file="/path/to/your/data.csv",
    output_dir="embeddings/custom-output",
)
```

## Supported Models

### Generative Models (vLLM)

| Model | HuggingFace ID | Parameters |
|-------|----------------|------------|
| Qwen3-Embedding-0.6B | `Qwen/Qwen3-Embedding-0.6B` | 0.6B |
| Qwen3-Embedding-4B | `Qwen/Qwen3-Embedding-4B` | 4B |
| Qwen3-Embedding-8B | `Qwen/Qwen3-Embedding-8B` | 8B |

### Discriminative Models (SentenceTransformers)

| Alias | HuggingFace ID |
|-------|----------------|
| `e5-large` | `intfloat/multilingual-e5-large` |
| `bert-base` | `google-bert/bert-base-uncased` |
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` |

You can also use any model from HuggingFace that is compatible with SentenceTransformers.

## Usage examples

### Example 1: Basic usage with defaults

```python
# Generative
from generative.embedder import run_embedding_pipeline
run_embedding_pipeline()

# Discriminative
from discriminative.embedder import run_embedding_pipeline
run_embedding_pipeline()
```

### Example 2: Custom configuration

```python
from discriminative.config import EmbeddingConfig, MODELS
from discriminative.embedder import run_embedding_pipeline

config = EmbeddingConfig(
    model_name=MODELS["bge-m3"],
    batch_size=256,
    input_file="/data/medical_terms.csv",
    output_dir="embeddings/bge-m3-output",
    cuda_visible_devices="0,1",
)

run_embedding_pipeline(config)
```

### Example 3: Using the embedder class directly

```python
from generative.embedder import QwenEmbedder

# Initialize
embedder = QwenEmbedder(
    model_name_or_path="Qwen/Qwen3-Embedding-4B",
    instruction="Encode medical terminology for semantic search",
)

# Encode sentences
sentences = ["diabetes mellitus", "hypertension", "myocardial infarction"]
embeddings = embedder.encode(sentences, dim=512)

print(f"Shape: {embeddings.shape}")  # (3, 512)

# For queries (with instruction formatting)
query_embedding = embedder.encode(
    "What is high blood pressure?",
    is_query=True,
    dim=512,
)

# Cleanup
embedder.stop()
```

```python
from discriminative.embedder import DiscriminativeEmbedder

# Initialize
embedder = DiscriminativeEmbedder(
    model_name_or_path="intfloat/multilingual-e5-large",
    device="cuda:0",
)

# Encode sentences
sentences = ["diabetes mellitus", "hypertension", "myocardial infarction"]
embeddings = embedder.encode(sentences, batch_size=32)

print(f"Shape: {embeddings.shape}")  # (3, 1024)
print(f"Embedding dim: {embedder.embedding_dimension}")
```

### Example 4: Processing multiple models

```python
from discriminative.config import EmbeddingConfig, MODELS
from discriminative.embedder import run_embedding_pipeline

models_to_run = ["e5-large", "bge-m3", "minilm-l6"]

for model_alias in models_to_run:
    print(f"\n{'='*50}")
    print(f"Processing with {model_alias}")
    print('='*50)
    
    config = EmbeddingConfig(
        model_name=MODELS[model_alias],
        input_file="/data/umls_terms.csv",
    )
    
    run_embedding_pipeline(config)
```

## Input data format

The input CSV file should have the following structure:

| term | CUI | ENG | FRE | ... | SNOMED | ICD10 | ... |
|------|-----|-----|-----|-----|--------|-------|-----|
| diabetes | ["C0011847"] | 1 | 0 | ... | 1 | 1 | ... |
| hypertension | ["C0020538"] | 1 | 1 | ... | 1 | 0 | ... |

- `word`: The medical term to embed
- `CUI`: List of UMLS Concept Unique Identifiers (as string representation of list)
- Binary columns (1/0): Indicate which vocabularies/ontologies the term belongs to

The pipeline will:
1. Filter rows where each binary column equals 1
2. Generate embeddings for those terms
3. Save results to separate `.bin` files per column

## Output format

Embeddings are saved as pickle files (`.bin`) containing a list of dictionaries:

```python
[
    {
        "term": "diabetes mellitus",
        "embedding": np.array([...]),  # numpy array
        "cuis": ["C0011847", "C0011849"],
    },
    ...
]
```

### Loading embeddings

```python
from discriminative.utils import load_embeddings

results = load_embeddings("embeddings/model-name/embedding_SNOMED_model-name.bin")

for item in results[:5]:
    print(f"Term: {item['term']}")
    print(f"CUIs: {item['cuis']}")
    print(f"Embedding shape: {item['embedding'].shape}")
```


