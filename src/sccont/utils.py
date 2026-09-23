"""Reproducibility and device helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and, if available, CUDA).

    All results in the scCont paper were generated with ``seed=42``.

    Parameters
    ----------
    seed
        Random seed.
    deterministic
        If True, also set ``torch.backends.cudnn.deterministic = True`` and
        ``torch.backends.cudnn.benchmark = False``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str | torch.device | None = None) -> torch.device:
    """Return a ``torch.device``; defaults to CUDA when available, else CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
