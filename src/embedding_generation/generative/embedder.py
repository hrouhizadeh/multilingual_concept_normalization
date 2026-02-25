# coding: utf-8
"""
Embedding generation using Qwen3 Embedding models with vLLM.

Supports Qwen-Embedding model family:
- Qwen/Qwen3-Embedding-0.6B
- Qwen/Qwen3-Embedding-4B
- Qwen/Qwen3-Embedding-8B
"""

from typing import List, Union, Optional
from datetime import datetime

import torch
from vllm import LLM, PoolingParams
from vllm.distributed.parallel_state import destroy_model_parallel

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


class QwenEmbedder:
    """
    Embedding generator using Qwen3 Embedding models via vLLM.
    
    Supports Matryoshka representations for flexible embedding dimensions.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        instruction: Optional[str] = None,
        max_length: int = 8192,
    ):
        """
        Initialize the Qwen embedding model.
        
        Args:
            model_name_or_path: Path or HuggingFace model name.
            instruction: Default instruction for query encoding.
            max_length: Maximum sequence length.
        """
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        
        self.instruction = instruction
        self.max_length = max_length
        self.model = LLM(
            model=model_name_or_path,
            hf_overrides={"is_matryoshka": True}
        )
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """
        Format a query with instruction for embedding.
        
        Args:
            task_description: Task description/instruction.
            query: The query text.
            
        Returns:
            Formatted instruction + query string.
        """
        if task_description is None:
            task_description = self.instruction
        return f"Instruct: {task_description}\nQuery:{query}"
    
    def encode(
        self,
        sentences: Union[List[str], str],
        is_query: bool = False,
        instruction: Optional[str] = None,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Generate embeddings for sentences.
        
        Args:
            sentences: Single sentence or list of sentences.
            is_query: Whether to apply instruction formatting.
            instruction: Custom instruction (uses default if None).
            dim: Embedding dimension (-1 for full dimension).
                 Supports Matryoshka dimensions: 128, 256, 512, 1024, etc.
            
        Returns:
            Tensor of embeddings with shape (num_sentences, dim).
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        if is_query:
            sentences = [
                self.get_detailed_instruct(instruction, sent)
                for sent in sentences
            ]
        
        pooling_params = PoolingParams(dimensions=dim) if dim > 0 else PoolingParams()
        output = self.model.embed(sentences, pooling_params=pooling_params)
        
        return torch.tensor([o.outputs.embedding for o in output])
    
    def stop(self) -> None:
        """Clean up model resources."""
        destroy_model_parallel()


def process_column(
    df,
    column: str,
    embedder: QwenEmbedder,
    config: EmbeddingConfig,
) -> None:
    """
    Process a single column: filter rows and generate embeddings.
    
    Args:
        df: Input DataFrame.
        column: Column name to filter on (where value == 1).
        embedder: QwenEmbedder instance.
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
        
        embeddings = embedder.encode(batch_words, dim=config.embedding_dim)
        
        for word, embedding, cui in zip(batch_words, embeddings, batch_cuis):
            results.append({
                "term": word,
                "embedding": embedding.numpy(),
                "cuis": cui,
            })
        
        processed = min(i + config.batch_size, len(words))
        print(f"    Processed {processed}/{len(words)} words", end="\r")
    
    print()  # New line after progress
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
    embedder = QwenEmbedder(
        model_name_or_path=config.model_name,
        instruction=config.instruction,
        max_length=config.max_length,
    )
    
    # Process columns
    print(f"\nProcessing {len(columns)} columns...")
    print(f"Batch size: {config.batch_size}\n")
    
    for i, column in enumerate(columns, 1):
        print(f"[{i}/{len(columns)}] Processing column: {column}")
        process_column(df, column, embedder, config)
        print()
    
    # Cleanup
    embedder.stop()
    
    # Final timing
    end_time = datetime.now()
    print("Done!")
    print_duration(start_time, end_time)


if __name__ == "__main__":
    # Run with default configuration
    run_embedding_pipeline()
