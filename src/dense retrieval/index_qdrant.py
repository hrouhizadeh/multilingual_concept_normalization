"""
Step 2: Index embeddings into Qdrant.

Reads per-category .bin embedding files produced by generate_embeddings.py,
creates a Qdrant collection with scalar quantization, and uploads all
vectors with their metadata (term, ontology, CUIs).
"""

import os
import pickle
from pathlib import Path

import requests
from tqdm import tqdm

import config


def get_embedding_size(file_path: Path) -> int:
    """Extract embedding dimension from first item in pickle file."""
    with open(file_path, "rb") as f:
        results = pickle.load(f)
    return len(results[0]["embedding"])


def load_embeddings(file_path: Path) -> list[dict]:
    """Load embeddings from pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def delete_collection(collection_name: str) -> None:
    """Delete existing Qdrant collection."""
    requests.delete(f"{config.qdrant_url}/collections/{collection_name}")


def create_collection(collection_name: str, embedding_dim: int) -> dict:
    """Create Qdrant collection with scalar quantization."""
    body = {
        "vectors": {
            "size": embedding_dim,
            "distance": config.qdrant_distance,
        },
        "optimizers_config": {
            "memmap_threshold": config.qdrant_memmap_threshold,
        },
        "quantization_config": {
            "scalar": {
                "type": config.qdrant_quantization_type,
                "quantile": config.qdrant_quantization_quantile,
                "always_ram": True,
            }
        },
        "hnsw_config": {"on_disk": True},
    }
    url = f"{config.qdrant_url}/collections/{collection_name}"
    response = requests.put(url, json=body)
    return response.json()


def upload_batch(
    collection_name: str, items: list[dict], ontology: str, start_id: int
) -> bool:
    """Upload a batch of points to Qdrant."""
    points = [
        {
            "id": start_id + idx,
            "vector": item["embedding"].tolist(),
            "payload": {
                "term": item["term"],
                "ontology": ontology,
                "cuis": item.get("cuis", []),
            },
        }
        for idx, item in enumerate(items)
    ]
    url = f"{config.qdrant_url}/collections/{collection_name}/points"
    response = requests.put(url, json={"points": points})
    return response.status_code == 200


def extract_ontology(bin_file: Path, llm_tag: str) -> str:
    """Extract ontology name from embedding filename."""
    return bin_file.stem.replace("embedding_", "").replace(f"_{llm_tag}", "")


def main():
    embeddings_dir = os.path.join(config.embedding_output_dir, config.model_save_name)
    collection_name = config.qdrant_collection_name
    llm_tag = config.model_save_name

    data_path = Path(embeddings_dir)
    bin_files = sorted(data_path.glob("*.bin"))

    if not bin_files:
        print(f"No embedding files found in {embeddings_dir}")
        return

    print(f"Found {len(bin_files)} embedding files in {embeddings_dir}")
    for i, f in enumerate(bin_files):
        print(f"  [{i}] {f.name}")

    # Detect embedding dimension
    embedding_dim = get_embedding_size(bin_files[0])
    print(f"\nEmbedding dimension: {embedding_dim}")

    # Check if collection already exists
    check = requests.get(f"{config.qdrant_url}/collections/{collection_name}")
    if check.status_code == 200:
        print(f"Collection '{collection_name}' already exists. Deleting and recreating...")
        delete_collection(collection_name)

    result = create_collection(collection_name, embedding_dim)
    print(f"Collection '{collection_name}' created: {result}")

    point_id = 0

    for bin_file in tqdm(bin_files, desc="Processing files"):
        ontology = extract_ontology(bin_file, llm_tag)
        embeddings = load_embeddings(bin_file)
        print(f"\n  Importing {len(embeddings)} vectors for ontology: {ontology}")

        for i in tqdm(
            range(0, len(embeddings), config.qdrant_upload_batch_size),
            desc=f"  Uploading {ontology}",
            leave=False,
        ):
            batch = embeddings[i : i + config.qdrant_upload_batch_size]
            success = upload_batch(collection_name, batch, ontology, point_id)
            if not success:
                print(f"  Error uploading batch {i}-{i + len(batch)} for {ontology}")
            point_id += len(batch)

    print(f"\nDone! Total points uploaded to '{collection_name}': {point_id}")


if __name__ == "__main__":
    main()
