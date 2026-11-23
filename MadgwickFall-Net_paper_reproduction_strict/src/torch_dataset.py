"""
torch.utils.data.Dataset wrapper around KFallDataset.build_windows() output.
Requires torch. Not run in the dev sandbox (no torch there) -- exercised on
the GPU server.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    BRANCHES = ("body_accel", "body_gyro", "global_accel", "global_gyro")

    def __init__(self, records, indices=None):
        self.records = records if indices is None else [records[i] for i in indices]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        inputs = {}
        for name in self.BRANCHES:
            arr = np.asarray(r[name], dtype=np.float32)  # (T,3)
            inputs[name] = torch.from_numpy(arr.T.copy())  # -> (3,T)
        label = torch.tensor(int(r["label"]), dtype=torch.long)
        return {
            "inputs": inputs,
            "label": label,
            "trial_id": r["trial_id"],
            "subject_id": r["subject_id"],
            "is_fall_trial": r["is_fall"],
        }


def collate_fn(batch):
    inputs = {}
    for name in WindowDataset.BRANCHES:
        inputs[name] = torch.stack([b["inputs"][name] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    trial_ids = [b["trial_id"] for b in batch]
    subject_ids = [b["subject_id"] for b in batch]
    is_fall = [b["is_fall_trial"] for b in batch]
    return {
        "inputs": inputs, "label": labels,
        "trial_id": trial_ids, "subject_id": subject_ids,
        "is_fall_trial": is_fall,
    }


def make_loader(records, indices, batch_size=64, shuffle=False, num_workers=4, training=False):
    from torch.utils.data import DataLoader, RandomSampler
    ds = WindowDataset(records, indices)
    # Keep sample ordering independent of worker startup, including on resume.
    sample_rng = torch.Generator().manual_seed(42)
    worker_rng = torch.Generator().manual_seed(42)
    sampler = RandomSampler(ds, generator=sample_rng) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=False, sampler=sampler,
                       generator=worker_rng,
                       num_workers=num_workers, collate_fn=collate_fn,
                       pin_memory=True, drop_last=training and len(ds) % batch_size == 1,
                       persistent_workers=num_workers > 0)
