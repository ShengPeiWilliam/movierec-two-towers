"""Dataset with negative sampling for two-tower model."""

import csv
import random
import torch
from torch.utils.data import Dataset

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH, NUM_NEGATIVES


class MovieDataset(Dataset):
    def __init__(self, test_ratio=0.2):
        # Read ratings.csv
        user_ids = []
        item_ids = []
        with open(DATA_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_ids.append(int(row["user_id"]))
                item_ids.append(int(row["item_id"]))

        # Build remap dicts (original ID → 0-indexed)
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))
        self.user2idx = {uid: i for i, uid in enumerate(unique_users)}
        self.item2idx = {iid: i for i, iid in enumerate(unique_items)}
        self.idx2user = {i: uid for uid, i in self.user2idx.items()}
        self.idx2item = {i: iid for iid, i in self.item2idx.items()}
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)

        # Build user → list of watched items (0-indexed)
        user_items = {}
        for uid, iid in zip(user_ids, item_ids):
            uidx = self.user2idx[uid]
            iidx = self.item2idx[iid]
            if uidx not in user_items:
                user_items[uidx] = []
            user_items[uidx].append(iidx)

        # Train / test split per user
        self.user_train = {}
        self.user_test = {}
        for uidx, items in user_items.items():
            random.shuffle(items)
            split = max(1, int(len(items) * (1 - test_ratio)))
            self.user_train[uidx] = set(items[:split])
            self.user_test[uidx] = set(items[split:])

        # Build user_watched = train only (for negative sampling)
        self.user_watched = self.user_train

        # Build positive + negative pairs from train set only
        self.samples = []
        all_items = set(range(self.num_items))

        for uidx, train_items in self.user_train.items():
            for iidx in train_items:
                # Positive
                self.samples.append((uidx, iidx, 1))
                # Negative
                unseen = list(all_items - train_items)
                negatives = random.sample(unseen, min(NUM_NEGATIVES, len(unseen)))
                for neg_iidx in negatives:
                    self.samples.append((uidx, neg_iidx, -1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uidx, iidx, label = self.samples[idx]
        return (
            torch.tensor(uidx, dtype=torch.long),
            torch.tensor(iidx, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )
