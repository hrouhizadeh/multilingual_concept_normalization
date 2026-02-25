import os
from pathlib import Path
from typing import Union

class Config:
    """Configuration settings for Elasticsearch and BM25 Search."""
    
    # Elasticsearch Settings
    ELASTIC_URL = os.getenv("ELASTIC_URL", "http://localhost:9400")
    INDEX_NAME = os.getenv("INDEX_NAME", "umls_concepts")
    TOP_K = 1000
    
    # Search Settings
    INPUT_FIELD = "term_all"
    APPLY_EXACT_MATCH_FILTER = True 
    FILTER_VALUE = 0 

    @staticmethod
    def setup_paths(input_dir: str, output_dir: str) -> tuple[Path, Path]:
        """
        Validates and returns Path objects for input and output.
        """
        input_path = Path(input_dir)
        output_base = Path(output_dir)
        save_path = output_base / f"dev_{Config.INDEX_NAME}"

        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")
            
        save_path.mkdir(parents=True, exist_ok=True)
        
        return input_path, save_path