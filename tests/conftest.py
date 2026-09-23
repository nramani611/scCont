import numpy as np
import pandas as pd
import pytest

from sccont import TemporalSingleCellDataset, set_seeds

N_GENES = 50
N_CELLS = 200
N_TIMEPOINTS = 4


@pytest.fixture(scope="session")
def toy_matrix() -> pd.DataFrame:
    """Genes x cells matrix with a smooth time signal in the first 10 genes."""
    set_seeds(0)
    rng = np.random.RandomState(0)
    t = np.repeat(np.arange(N_TIMEPOINTS), N_CELLS // N_TIMEPOINTS)
    X = rng.normal(size=(N_GENES, N_CELLS)).astype(np.float32)
    X[:10] += t[None, :] * 1.5
    genes = [f"G{i}" for i in range(N_GENES)]
    cells = [f"V{i}_T{ti}" for i, ti in enumerate(t)]
    return pd.DataFrame(X, index=genes, columns=cells)


@pytest.fixture(scope="session")
def toy_timepoints(toy_matrix):
    return [c.split("_")[1] for c in toy_matrix.columns]


@pytest.fixture(scope="session")
def toy_dataset(toy_matrix, toy_timepoints):
    return TemporalSingleCellDataset(toy_matrix.to_numpy(), toy_timepoints)
