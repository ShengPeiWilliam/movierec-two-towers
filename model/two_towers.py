"""Two-tower model: User Tower + Item Tower."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_DIM


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
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, item_idx):
        x = self.embedding(item_idx)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, dim=EMBEDDING_DIM):
        super().__init__()
        self.user_tower = UserTower(num_users, dim)
        self.item_tower = ItemTower(num_items, dim)

    def forward(self, user_idx, item_idx):
        user_emb = self.user_tower(user_idx)
        item_emb = self.item_tower(item_idx)
        return user_emb, item_emb
