"""Evaluate retrieval quality using Recall@K across multiple K values."""

import os
import torch
import chromadb

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR, CHROMA_DIR, COLLECTION_NAME
from model.dataset import MovieDataset
from model.two_towers import TwoTowerModel

K_VALUES = [10, 20, 50]


def evaluate():
    # Load dataset (with train/test split)
    print("Loading dataset...")
    dataset = MovieDataset(test_ratio=0.2)

    # Load model
    print("Loading model...")
    model = TwoTowerModel(dataset.num_users, dataset.num_items)
    model.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, "model.pt"), weights_only=True)
    )
    model.eval()

    # Connect to Chroma
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    # Evaluate Recall@K for each user who has test items
    max_k = max(K_VALUES)
    recall_scores = {k: [] for k in K_VALUES}

    users_with_test = [
        uidx for uidx, test_items in dataset.user_test.items()
        if len(test_items) > 0
    ]

    print(f"Evaluating {len(users_with_test)} users...")

    for uidx in users_with_test:
        test_items = dataset.user_test[uidx]  # set of item idx

        # Get user embedding
        user_idx_tensor = torch.tensor(uidx, dtype=torch.long)
        with torch.no_grad():
            user_emb = model.user_tower(user_idx_tensor.unsqueeze(0))
        query = user_emb.squeeze(0).tolist()

        # Query Chroma for top max_k results
        results = collection.query(
            query_embeddings=[query],
            n_results=max_k,
            include=["metadatas"],
        )
        recommended_ids = results["ids"][0]  # list of item_id strings

        # Convert recommended item_ids back to idx
        recommended_idx = {
            dataset.item2idx[int(iid)]
            for iid in recommended_ids
            if int(iid) in dataset.item2idx
        }

        # Compute Recall@K
        for k in K_VALUES:
            top_k = set(list(recommended_idx)[:k])
            hits = len(top_k & test_items)
            recall = hits / len(test_items)
            recall_scores[k].append(recall)

    # Print results
    print("\n── Recall@K Results ──")
    for k in K_VALUES:
        avg = sum(recall_scores[k]) / len(recall_scores[k])
        print(f"  Recall@{k:<3} = {avg:.4f}")


if __name__ == "__main__":
    evaluate()
