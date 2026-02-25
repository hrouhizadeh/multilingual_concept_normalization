# coding: utf-8
"""
Configuration settings for UMLS embedding generation with Qwen-Embedding models.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation with generative models."""
    
    # Model settings
    model_name: str = "Qwen/Qwen3-Embedding-8B"
    embedding_dim: int = 1024
    max_length: int = 8192
    instruction: str = "Given a web search query, retrieve relevant passages that answer the query"
    
    # Processing settings
    batch_size: int = 100
    save_every: int = 10000
    
    # File paths
    input_file: str = "Path/ to/ input/ csv"
    output_dir: str = ""
    
    # Columns to exclude from processing
    exclude_columns: List[str] = field(default_factory=lambda: [
        "word", "CUI", "ENG", "FRE", "GER", "SPA", "TUR"
    ])
    
    def __post_init__(self):
        """Set derived attributes after initialization."""
        self.model_save_name = self.model_name.replace("/", "-").replace("_", "-")
        if not self.output_dir:
            self.output_dir = f"embeddings/{self.model_save_name}"
        os.makedirs(self.output_dir, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = EmbeddingConfig()


# Supported Qwen-Embedding models
MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
}
