from __future__ import annotations

from torch.utils.data._utils.collate import default_collate


def collate_tokenized_batch(batch):
    """Collate fixed-length tokenized examples produced by TokenizedTextDataset."""
    return default_collate(batch)

