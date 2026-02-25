# coding: utf-8
"""
Configuration settings for UMLS embedding generation with discriminative models.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation with discriminative models."""
    
    # Model settings
    model_name: str = "intfloat/multilingual-e5-large"
    device: Optional[str] = None  # None for auto-detect, or "cuda:0", "cpu", etc.
    
    # Processing settings
    batch_size: int = 500
    save_every: int = 10000
    show_progress_bar: bool = False
    
    # File paths
    input_file: str = "Path/ to/ input/ csv"
    output_dir: str = ""
    
    # Columns to exclude from processing
    exclude_columns: List[str] = field(default_factory=lambda: [
        "word", "CUI", "ENG", "FRE", "GER", "SPA", "TUR"
    ])
    
    # CUDA settings
    cuda_visible_devices: Optional[str] = "0"
    
    def __post_init__(self):
        """Set derived attributes after initialization."""
        # Set CUDA devices if specified
        if self.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        
        self.model_save_name = self.model_name.replace("/", "-").replace("_", "-")
        if not self.output_dir:
            self.output_dir = f"embeddings/{self.model_save_name}"
        os.makedirs(self.output_dir, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = EmbeddingConfig()


# Common model presets
MODELS = {
    # Multilingual E5 models
    "e5-large": "intfloat/multilingual-e5-large",
    "e5-base": "intfloat/multilingual-e5-base",
    "e5-small": "intfloat/multilingual-e5-small",
    
    # BERT models
    "bert-large": "google-bert/bert-large-uncased",
    "bert-base": "google-bert/bert-base-uncased",
    
    # MPNet
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}
