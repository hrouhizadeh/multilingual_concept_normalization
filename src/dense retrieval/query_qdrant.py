"""
Step 3: Query Qdrant for candidate CUI retrieval.

For each input CSV (per dataset split), encodes query terms using the same
Llama-Embed-Nemotron-8B model, searches the Qdrant collection filtered by
source ontology, applies semantic group filtering, and saves ranked results.
"""

import os
import ast
import json
import time
from datetime import datetime

import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

import config


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model() -> SentenceTransformer:
    """Load the same embedding model used for indexing."""
    return SentenceTransformer(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        model_kwargs={
            "attn_implementation": config.attention_implementation,
            "torch_dtype": config.torch_dtype,
        },
        tokenizer_kwargs={"padding_side": config.padding_side},
    )


def encode_queries(model: SentenceTransformer, queries: list[str]) -> list[list[float]]:
    """Encode query strings to embedding vectors."""
    return model.encode(queries).tolist()


# ---------------------------------------------------------------------------
# Qdrant search
# ---------------------------------------------------------------------------

def search_batch(
    model: SentenceTransformer,
    queries: list[str],
    top_k: int,
    ontology_list: list[str] | None = None,
) -> list[list[dict]]:
    """Search collection with multiple query vectors at once."""
    url = f"{config.qdrant_url}/collections/{config.qdrant_collection_name}/points/search/batch"

    search_request = {
        "vector": None,
        "limit": top_k,
        "with_payload": True,
    }

    if ontology_list:
        search_request["filter"] = {
            "should": [
                {"key": "ontology", "match": {"value": ont}} for ont in ontology_list
            ]
        }

    query_vectors = encode_queries(model, queries)
    valid_vectors = [(i, vec) for i, vec in enumerate(query_vectors) if vec is not None]

    if not valid_vectors:
        return [[] for _ in query_vectors]

    payload = {
        "searches": [{**search_request, "vector": vec} for _, vec in valid_vectors]
    }

    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        return [[] for _ in query_vectors]

    batch_results = response.json().get("result", [])
    results = [[] for _ in query_vectors]
    for (orig_idx, _), result in zip(valid_vectors, batch_results):
        results[orig_idx] = result

    return results


def search(
    model: SentenceTransformer,
    queries: list[str],
    top_k: int,
    ontology_list: list[str] | None = None,
) -> list[list[dict]]:
    """Search queries in batches to avoid overloading the server."""
    all_results = []
    total = len(queries)
    bs = config.query_batch_size

    print(f"Processing {total} queries in batches of {bs}...")

    for i in range(0, total, bs):
        batch_end = min(i + bs, total)
        batch_queries = queries[i:batch_end]
        print(
            f"  Batch {i // bs + 1}/{(total + bs - 1) // bs} "
            f"(queries {i + 1}-{batch_end})"
        )
        batch_results = search_batch(model, batch_queries, top_k, ontology_list)
        all_results.extend(batch_results)

        if batch_end < total:
            time.sleep(0.1)

    return all_results


# ---------------------------------------------------------------------------
# Result formatting with semantic group filtering
# ---------------------------------------------------------------------------

def format_results(
    query_sem_groups: list,
    results: list[dict],
    cui_semantic_mapping: dict,
    top_k_save: int,
) -> tuple[list[str], list[float]]:
    """Filter and rank retrieved CUIs by semantic group overlap."""
    query_sem_set = set(query_sem_groups)

    if not results:
        return [], []

    final_cuis = []
    final_scores = []
    seen_codes = set()

    for hit in results:
        cui_codes = hit["payload"]["cuis"]
        score = hit["score"]
        for cui in cui_codes:
            if cui in seen_codes:
                continue
            seen_codes.add(cui)
            cui_info = cui_semantic_mapping.get(cui, {})
            cui_sem_groups = set(cui_info.get("sem_group", []))
            if query_sem_set & cui_sem_groups:
                final_cuis.append(cui)
                final_scores.append(score)

    return final_cuis[:top_k_save], final_scores[:top_k_save]


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_file(
    input_file: str,
    output_dir: str,
    model: SentenceTransformer,
    cui_semantic_mapping: dict,
    target_ontologies: list[str],
    time_str: str,
):
    """Process a single CSV file: encode, search, format, save."""
    df = pd.read_csv(input_file)
    queries = [s.lower() for s in df["term"].tolist()]
    semantic_groups = df["semantic_group"].apply(ast.literal_eval)

    results = search(
        model, queries, top_k=config.query_top_k_fetch, ontology_list=target_ontologies
    )

    retrieved_codes = []
    retrieved_scores = []
    for query, sem_group, result in zip(queries, semantic_groups, results):
        codes, scores = format_results(
            sem_group, result, cui_semantic_mapping, config.query_top_k_save
        )
        retrieved_codes.append(codes)
        retrieved_scores.append(scores)

    df["retrieved_codes"] = retrieved_codes
    df["retrieved_scores"] = retrieved_scores

    # Move embedding column to end if present
    if "embedding" in df.columns:
        cols = [c for c in df.columns if c != "embedding"] + ["embedding"]
        df = df[cols]

    base = os.path.basename(input_file)
    out_name = base.replace(
        ".csv",
        f"_top{config.query_top_k_save}_{config.qdrant_collection_name}_{time_str}_result.csv",
    )
    out_path = os.path.join(output_dir, out_name)
    df.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load CUI semantic mapping
    print(f"Loading CUI semantic mapping from: {config.cui_semantic_mapping_path}")
    with open(config.cui_semantic_mapping_path, "r") as f:
        cui_semantic_mapping = json.load(f)

    # Load model (same as used for indexing)
    print(f"Loading model: {config.model_name}")
    model = load_model()

    for split in config.query_splits:
        split_path = os.path.join(config.query_data_dir, split)
        output_dir = os.path.join(
            config.query_output_dir, split, config.model_save_name
        )
        os.makedirs(output_dir, exist_ok=True)

        csv_files = sorted(
            [f for f in os.listdir(split_path) if f.endswith(".csv")]
        )

        print(f"\n{'=' * 60}")
        print(f"Split: {split} — {len(csv_files)} files")
        print(f"{'=' * 60}")

        for csv_f in csv_files:
            print(f"\nProcessing: {csv_f}")
            input_file = os.path.join(split_path, csv_f)
            dataset_name = csv_f.split("_")[0]
            target_ontologies = config.source_ontology_dict.get(dataset_name, [])

            if not target_ontologies:
                print(f"  WARNING: No ontology mapping for dataset '{dataset_name}', skipping.")
                continue

            process_file(
                input_file, output_dir, model, cui_semantic_mapping,
                target_ontologies, time_str,
            )

    print(f"\n{'=' * 60}")
    print("All splits processed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
