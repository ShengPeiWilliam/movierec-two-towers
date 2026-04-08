"""Two-tower model: User Tower + Item Tower (with genre embedding)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_DIM, NUM_GENRES, GENRE_PADDING_IDX


class UserTower(nn.Module):
    def __init__(self, num_users, dim=EMBEDDING_DIM):
        super().__init__()
        self.embedding = nn.Embedding(num_users, dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, user_idx):
        x = self.embedding(user_idx)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1)


class ItemTower(nn.Module):
    def __init__(self, num_items, dim=EMBEDDING_DIM):
        super().__init__()
        self.embedding = nn.Embedding(num_items, dim)
        # Genre embedding: NUM_GENRES + 1 (padding)，padding_idx 不更新
        self.genre_embedding = nn.Embedding(NUM_GENRES + 1, dim, padding_idx=GENRE_PADDING_IDX)
        # Input = item_emb (dim) + genre_emb (dim) = 2*dim
        self.fc1 = nn.Linear(dim * 2, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, item_idx, genre_idx):
        # item_idx: [batch]
        # genre_idx: [batch, MAX_GENRE_LEN]
        item_emb = self.embedding(item_idx)                          # [batch, dim]

        genre_embs = self.genre_embedding(genre_idx)                 # [batch, MAX_GENRE_LEN, dim]
        mask = (genre_idx != GENRE_PADDING_IDX).float()              # [batch, MAX_GENRE_LEN]
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)        # [batch, 1]
        genre_emb = (genre_embs * mask.unsqueeze(-1)).sum(dim=1)     # [batch, dim]
        genre_emb = genre_emb / mask_sum                             # mean pooling

        x = torch.cat([item_emb, genre_emb], dim=-1)                 # [batch, dim*2]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, dim=EMBEDDING_DIM):
        super().__init__()
        self.user_tower = UserTower(num_users, dim)
        self.item_tower = ItemTower(num_items, dim)

    def forward(self, user_idx, item_idx, genre_idx):
        user_emb = self.user_tower(user_idx)
        item_emb = self.item_tower(item_idx, genre_idx)
        return user_emb, item_emb
