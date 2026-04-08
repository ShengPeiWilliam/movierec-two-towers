import os
import csv
import torch
import chromadb

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR, CHROMA_DIR, COLLECTION_NAME, MOVIES_PATH


def load_movies(movies_path):
    """Read movies.csv → {item_id: {title, genre}}"""
    movies = {}
    with open(movies_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[int(row["item_id"])] = {
                "title": row["title"],
                "genre": row["genre"],
            }
    return movies


def embed_items():
    # Load checkpoints
    print("Loading checkpoints...")
    item_embs = torch.load(os.path.join(CHECKPOINT_DIR, "item_embeddings.pt"), weights_only=True)
    mappings = torch.load(os.path.join(CHECKPOINT_DIR, "mappings.pt"), weights_only=True)
    idx2item = mappings["idx2item"]

    # Load movie metadata
    movies = load_movies(MOVIES_PATH)

    # Setup Chroma
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if exists
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare data
    embeddings = []
    ids = []
    metadatas = []

    for idx, emb in enumerate(item_embs):
        item_id = idx2item[idx]
        movie = movies.get(item_id, {"title": "Unknown", "genre": "unknown"})

        embeddings.append(emb.tolist())
        ids.append(str(item_id))
        metadatas.append({
            "item_id": item_id,
            "title": movie["title"],
            "genre": movie["genre"],
        })

    # Store into Chroma
    collection.add(
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"Stored {len(ids)} items into ChromaDB collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    embed_items()
