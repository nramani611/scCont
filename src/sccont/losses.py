"""InfoNCE (NT-Xent) contrastive loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE loss over a batch of positive pairs ``(z_i[b], z_j[b])``.

    For each of the ``2B`` embeddings the single positive is its paired view;
    the remaining ``2B - 2`` embeddings in the batch act as negatives.

    Parameters
    ----------
    temperature
        Softmax temperature (the paper uses 0.20).
    """

    def __init__(self, temperature: float = 0.20):
        super().__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=2)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)

        z = torch.cat([z_i, z_j], dim=0)  # (2B, proj_dim)
        sim = self.cosine_sim(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature  # (2B, 2B)

        # Positive mask: the off-diagonal blocks' diagonals.
        mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        mask = mask.repeat(2, 2)
        mask = mask.fill_diagonal_(False)

        pos = sim[mask].view(2 * batch_size, -1)
        neg = sim[~mask].view(2 * batch_size, -1)

        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        return F.cross_entropy(logits, labels)
