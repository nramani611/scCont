"""Encoder and projector networks used for contrastive training."""

from __future__ import annotations

import torch.nn as nn


class Encoder(nn.Module):
    """MLP encoder mapping gene expression to latent features (LFs).

    Architecture: ``input_dim -> 2048 -> 1024 -> 64 -> latent_dim`` with
    BatchNorm + ReLU between layers. The latent layer is linear so LF activations
    can be positive or negative.
    """

    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Projector(nn.Module):
    """Projection head applied on top of the encoder during training only."""

    def __init__(self, latent_dim: int = 32, proj_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, proj_dim),
        )

    def forward(self, z):
        return self.net(z)
