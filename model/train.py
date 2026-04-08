"""Train two-tower model and save checkpoints."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR, EMBEDDING_DIM, BATCH_SIZE, EPOCHS, LR
from model.dataset import MovieDataset
from model.two_towers import TwoTowerModel


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Loading dataset...")
    dataset = MovieDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Users: {dataset.num_users}, Items: {dataset.num_items}")
    print(f"Total samples: {len(dataset)}")

    model = TwoTowerModel(dataset.num_users, dataset.num_items, EMBEDDING_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CosineEmbeddingLoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for user_idx, item_idx, genre_idx, label in loader:
            user_emb, item_emb = model(user_idx, item_idx, genre_idx)
            loss = loss_fn(user_emb, item_emb, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{EPOCHS}  loss: {avg_loss:.4f}")

    # Save item embeddings (need genre tensors for all items)
    print("Saving checkpoints...")
    model.eval()
    with torch.no_grad():
        all_item_idx = torch.arange(dataset.num_items)
        all_genre_idx = torch.stack([
            dataset.get_genre(i) for i in range(dataset.num_items)
        ])
        item_embs = model.item_tower(all_item_idx, all_genre_idx)
        torch.save(item_embs, os.path.join(CHECKPOINT_DIR, "item_embeddings.pt"))

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model.pt"))
    torch.save({
        "user2idx": dataset.user2idx,
        "item2idx": dataset.item2idx,
        "idx2item": dataset.idx2item,
        "idx2user": dataset.idx2user,
        "item2genre": {i: dataset.get_genre(i) for i in range(dataset.num_items)},
    }, os.path.join(CHECKPOINT_DIR, "mappings.pt"))

    print("Done! Checkpoints saved to", CHECKPOINT_DIR)


if __name__ == "__main__":
    train()
