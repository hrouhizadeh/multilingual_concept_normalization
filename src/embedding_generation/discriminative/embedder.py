# coding: utf-8
"""
Embedding generation using discriminative language models (SentenceTransformers).
Supports any model compatible with the sentence-transformers library
"""

from typing import List, Union, Optional
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EmbeddingConfig, DEFAULT_CONFIG, MODELS
from utils import (
    load_csv,
    parse_cui_column,
    get_processing_columns,
    filter_by_column,
    save_embeddings,
    print_timing,
    print_duration,
)


class DiscriminativeEmbedder:
    """
    Embedding generator using discriminative language models via SentenceTransformers.
    
    Supports any model compatible with the sentence-transformers library.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        show_progress_bar: bool = False,
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name_or_path: Path or HuggingFace model name.
            device: Device to use (None for auto-detect).
            show_progress_bar: Whether to show progress during encoding.
        """
        self.model_name = model_name_or_path
        self.show_progress_bar = show_progress_bar
        self.model = SentenceTransformer(model_name_or_path, device=device)
    
    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 32,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for sentences.
        
        Args:
            sentences: Single sentence or list of sentences.
            batch_size: Batch size for encoding.
            normalize: Whether to normalize embeddings.
            
        Returns:
            Numpy array of embeddings with shape (num_sentences, dim).
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=normalize,
        )
        
        return embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()


def process_column(
    df,
    column: str,
    embedder: DiscriminativeEmbedder,
    config: EmbeddingConfig,
) -> None:
    """
    Process a single column: filter rows and generate embeddings.
    
    Args:
        df: Input DataFrame.
        column: Column name to filter on (where value == 1).
        embedder: DiscriminativeEmbedder instance.
        config: Configuration object.
    """
    words, cuis = filter_by_column(df, column)
    
    if len(words) == 0:
        print(f"  No words found for {column}, skipping...")
        return
    
    print(f"  Filtered rows ({column}=1): {len(words)}")
    
    output_file = f"{config.output_dir}/embedding_{column}_{config.model_save_name}.bin"
    results = []
    
    for i in range(0, len(words), config.batch_size):
        batch_words = words[i:i + config.batch_size]
        batch_cuis = cuis[i:i + config.batch_size]
        
        embeddings = embedder.encode(batch_words, batch_size=config.batch_size)
        
        for word, embedding, cui in zip(batch_words, embeddings, batch_cuis):
            results.append({
                "term": word,
                "embedding": embedding,
                "cuis": cui,
            })
    
    save_embeddings(results, output_file)


def run_embedding_pipeline(config: Optional[EmbeddingConfig] = None) -> None:
    """
    Run the full embedding generation pipeline.
    
    Args:
        config: Configuration object (uses DEFAULT_CONFIG if None).
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Timing
    start_time = datetime.now()
    print_timing(start_time, "Start")
    print("=" * 50)
    
    # Load and prepare data
    df = load_csv(config.input_file)
    df = parse_cui_column(df, "CUI")
    columns = get_processing_columns(df, config.exclude_columns)
    
    # Initialize model
    print(f"\nLoading model: {config.model_name}")
    embedder = DiscriminativeEmbedder(
        model_name_or_path=config.model_name,
        device=config.device,
        show_progress_bar=config.show_progress_bar,
    )
    print(f"Embedding dimension: {embedder.embedding_dimension}")
    
    # Process columns
    print(f"\nProcessing {len(columns)} columns...")
    print(f"Batch size: {config.batch_size}\n")
    
    for i, column in enumerate(columns, 1):
        print(f"[{i}/{len(columns)}] Processing column: {column}")
        process_column(df, column, embedder, config)
        print()
    
    # Final timing
    end_time = datetime.now()
    print("Done!")
    print_duration(start_time, end_time)


if __name__ == "__main__":
    # Run with default configuration
    run_embedding_pipeline()
