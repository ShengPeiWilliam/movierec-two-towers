import csv
import random
import torch
from torch.utils.data import Dataset

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH, NUM_NEGATIVES


class MovieDataset(Dataset):
    def __init__(self):
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

        # Build user → set of watched items (0-indexed)
        self.user_watched = {}
        for uid, iid in zip(user_ids, item_ids):
            uidx = self.user2idx[uid]
            iidx = self.item2idx[iid]
            if uidx not in self.user_watched:
                self.user_watched[uidx] = set()
            self.user_watched[uidx].add(iidx)

        # Build positive + negative pairs
        self.samples = []
        all_items = set(range(self.num_items))

        for uid, iid in zip(user_ids, item_ids):
            uidx = self.user2idx[uid]
            iidx = self.item2idx[iid]
            # Positive: user watched this item
            self.samples.append((uidx, iidx, 1))
            # Negative: random items user has NOT watched
            unseen = list(all_items - self.user_watched[uidx])
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
