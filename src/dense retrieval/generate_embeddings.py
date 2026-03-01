#!/usr/bin/env python3
"""
Step 1: Generate UMLS concept embeddings.

Reads the UMLS CSV, filters rows per semantic category column,
encodes terms with Llama-Embed-Nemotron-8B, and saves per-category
embedding files as serialized pickle (.bin).

Usage:
    python generate_embeddings.py
"""

import ast
import logging
import os
import pickle
import time
from datetime import datetime

import pandas as pd
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def filter_and_embed_column(
    df: pd.DataFrame,
    column: str,
    model: SentenceTransformer,
    model_save_name: str,
    output_dir: str,
    batch_size: int,
) -> None:
    """Filter rows where *column* == 1, encode terms, and persist embeddings.

    Each output file is a pickled list of dicts with keys
    ``{"term", "embedding", "cuis"}``.

    Parameters
    ----------
    df : pd.DataFrame
        Full UMLS DataFrame (must contain ``term`` and ``CUI`` columns).
    column : str
        Binary semantic-category column to filter on.
    model : SentenceTransformer
        Pre-loaded embedding model.
    model_save_name : str
        Short model identifier used in the output filename.
    output_dir : str
        Directory where ``.bin`` files are written.
    batch_size : int
        Number of terms encoded per forward pass.
    """
    filtered_df = df[df[column] == 1]
    terms = filtered_df["term"].str.lower().tolist()
    cuis = filtered_df["CUI"].tolist()

    if not terms:
        logger.info("  No terms found for '%s' — skipping.", column)
        return

    logger.info("  Filtered rows (%s=1): %d", column, len(terms))

    results: list[dict] = []
    for i in range(0, len(terms), batch_size):
        batch_terms = terms[i : i + batch_size]
        batch_cuis = cuis[i : i + batch_size]
        embeddings = model.encode(batch_terms, show_progress_bar=False)

        for term, embedding, cui in zip(batch_terms, embeddings, batch_cuis):
            results.append({"term": term, "embedding": embedding, "cuis": cui})

    output_file = os.path.join(output_dir, f"embedding_{column}_{model_save_name}.bin")
    with open(output_file, "wb") as fh:
        pickle.dump(results, fh)

    logger.info("  Saved %d items → %s", len(results), output_file)


def load_model(model_name: str) -> SentenceTransformer:
    """Load and return the SentenceTransformer embedding model."""
    logger.info("Loading model: %s", model_name)
    return SentenceTransformer(
        model_name,
        trust_remote_code=config.trust_remote_code,
        model_kwargs={
            "attn_implementation": config.attention_implementation,
            "torch_dtype": config.torch_dtype,
        },
        tokenizer_kwargs={"padding_side": config.padding_side},
    )


def load_umls_dataframe(path: str) -> pd.DataFrame:
    """Read the UMLS CSV and parse the CUI column from string to list."""
    logger.info("Reading CSV: %s", path)
    df = pd.read_csv(path)
    logger.info("Total rows: %d", len(df))

    if "CUI" not in df.columns:
        raise ValueError("Required column 'CUI' not found in the CSV file.")

    logger.info("Parsing 'CUI' column (string → list)…")
    df["CUI"] = df["CUI"].apply(ast.literal_eval)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("UMLS Embedding Generation — started %s", datetime.now().isoformat())
    logger.info("=" * 60)

    # Prepare output directory
    output_dir = os.path.join(config.embedding_output_dir, config.model_save_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_umls_dataframe(config.embedding_input_file)

    # Determine semantic-category columns to process
    columns = [c for c in df.columns if c not in config.embedding_exclude_columns]
    logger.info("Excluded columns : %s", config.embedding_exclude_columns)
    logger.info("Target columns   : %s", columns)

    # Load model
    model = load_model(config.model_name)

    # Process each semantic category
    logger.info("Processing %d columns (batch_size=%d)…", len(columns), config.embedding_batch_size)
    for idx, column in enumerate(columns, 1):
        logger.info("[%d/%d] Column: %s", idx, len(columns), column)
        filter_and_embed_column(
            df=df,
            column=column,
            model=model,
            model_save_name=config.model_save_name,
            output_dir=output_dir,
            batch_size=config.embedding_batch_size,
        )

    elapsed = time.perf_counter() - start
    logger.info("=" * 60)
    logger.info("Finished in %.1f s (%.1f min)", elapsed, elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
