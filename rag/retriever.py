"""Retrieve top-k movie recommendations for a given user."""

import os
import torch
import chromadb

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR, CHROMA_DIR, COLLECTION_NAME
from model.two_towers import TwoTowerModel


class Retriever:
    def __init__(self):
        print("Loading mappings...")
        mappings = torch.load(os.path.join(CHECKPOINT_DIR, "mappings.pt"), weights_only=True)
        self.user2idx = mappings["user2idx"]
        self.idx2item = mappings["idx2item"]

        print("Loading model...")
        # Infer num_users and num_items from mappings
        num_users = len(self.user2idx)
        num_items = len(self.idx2item)
        self.model = TwoTowerModel(num_users, num_items)
        self.model.load_state_dict(
            torch.load(os.path.join(CHECKPOINT_DIR, "model.pt"), weights_only=True)
        )
        self.model.eval()

        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)

        # Fallback: popular items (by item_id order, first 10)
        self._fallback = self._get_fallback()

    def _get_fallback(self):
        """Return top-10 most popular items as fallback for unseen users."""
        results = self.collection.get(limit=10, include=["metadatas"])
        return results["metadatas"]

    def recommend(self, user_id: int, k: int = 10):
        # Unseen user fallback
        if user_id not in self.user2idx:
            return self._fallback[:k]

        # Get user embedding
        user_idx = torch.tensor(self.user2idx[user_id], dtype=torch.long)
        with torch.no_grad():
            user_emb = self.model.user_tower(user_idx.unsqueeze(0))
        query = user_emb.squeeze(0).tolist()

        # Query Chroma
        results = self.collection.query(
            query_embeddings=[query],
            n_results=k,
            include=["metadatas"],
        )
        return results["metadatas"][0]
