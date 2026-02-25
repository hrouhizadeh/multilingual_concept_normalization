# coding: utf-8
"""
Utility functions for UMLS embedding generation.
"""

import ast
import pickle
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        Loaded DataFrame.
    """
    print(f"Reading CSV file: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Total rows in file: {len(df)}")
    return df


def parse_cui_column(df: pd.DataFrame, column: str = "CUI") -> pd.DataFrame:
    """
    Parse a string column containing list representations to actual lists.
    
    Args:
        df: Input DataFrame.
        column: Column name to parse.
        
    Returns:
        DataFrame with parsed column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the CSV file!")
    
    print(f"\nParsing '{column}' column from string to list...")
    df[column] = df[column].apply(ast.literal_eval)
    return df


def get_processing_columns(df: pd.DataFrame, exclude_columns: List[str]) -> List[str]:
    """
    Get list of columns to process (excluding specified columns).
    
    Args:
        df: Input DataFrame.
        exclude_columns: List of column names to exclude.
        
    Returns:
        List of column names to process.
    """
    all_columns = df.columns.tolist()
    columns = [col for col in all_columns if col not in exclude_columns]
    
    print(f"\nExcluded columns: {exclude_columns}")
    print(f"Processing columns: {columns}")
    
    return columns


def filter_by_column(df: pd.DataFrame, column: str) -> Tuple[List[str], List[Any]]:
    """
    Filter DataFrame rows where specified column equals 1.
    
    Args:
        df: Input DataFrame.
        column: Column name to filter on.
        
    Returns:
        Tuple of (words list, cuis list).
    """
    filtered_df = df[df[column] == 1].copy()
    words = filtered_df["word"].str.lower().tolist()
    cuis = filtered_df["CUI"].tolist()
    return words, cuis


def save_embeddings(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save embedding results to a pickle file.
    
    Args:
        results: List of dictionaries containing term, embedding, and cuis.
        output_file: Path to output file.
    """
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved {len(results)} items to: {output_file}")


def load_embeddings(input_file: str) -> List[Dict[str, Any]]:
    """
    Load embeddings from a pickle file.
    
    Args:
        input_file: Path to input file.
        
    Returns:
        List of dictionaries containing term, embedding, and cuis.
    """
    with open(input_file, "rb") as f:
        return pickle.load(f)


def print_timing(start_time: datetime, message: str = "Start") -> None:
    """
    Print timing information.
    
    Args:
        start_time: Start datetime.
        message: Message prefix.
    """
    print(f"{message} time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")


def print_duration(start_time: datetime, end_time: datetime) -> None:
    """
    Print duration between two timestamps.
    
    Args:
        start_time: Start datetime.
        end_time: End datetime.
    """
    duration = end_time - start_time
    print("=" * 50)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
