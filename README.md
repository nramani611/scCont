# scCont

[![PyPI](https://img.shields.io/pypi/v/sccont)](https://pypi.org/project/sccont/)
[![CI](https://github.com/nramani611/scCont/actions/workflows/ci.yml/badge.svg)](https://github.com/nramani611/scCont/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`scCont` is an unsupervised contrastive deep learning model that extracts interpretable latent features from single-cell RNA data.

Contrastive learning is typically used to find similarities between actual and perturbed data, but we modify the typical contrastive learning framework to use the framework to learn local neighborhood functional group information in scRNA data. scCont achieves this through five steps: 1) _k_NN Pair Selection, 2) Contrastive Training, 3) Signal Identification via Spatial Clustering, 4) Network Attribution Analysis, 5) Biological Annotation. Figure 1 visualizes these steps.

![scCont Pipeline](https://raw.githubusercontent.com/nramani611/scCont/main/Figure1_V2.png)
The framework operates in five stages to extract interpretable latent features (LFs) from timecourse scRNA-seq data. **A) _k_-NN Pair Selection**: A _k_-Nearest Neighbor graph (K=3) is applied to the normalized gene expression space to identify positive pairs, preserving local topology. PCA is used here to reduce noise in the data. **B) Contrastive Training:** These pairs serve as inputs to a contrastive learning architecture that minimizes InfoNCE loss to maximize agreement between neighboring cells in the latent space. **C) Signal Identification via Spatial Clustering:** LF activations are projected onto the gene expression PCA space; features with correlated spatial patterns are grouped to isolate distinct signal trajectories. **D) Network Attribution Analysis:** Specific genes driving each LF are identified using feature importance scores (SHAP). **E) Biological Annotations:** LFs are connected to biological processes via a multi-tiered validation approach, combining broad Gene Ontology (GO) enrichment for clusters with functional group identification for individual top-contributing genes.

## Installation

```bash
pip install sccont
```

One install covers every stage, including preprocessing (Scrublet) and GO enrichment (g:Profiler).
Requires Python 3.10+. `pip` installs a CPU build of PyTorch by default; for GPU training install
torch first following <https://pytorch.org/get-started/locally/>, then install `sccont`.

To work from a source checkout instead:

```bash
git clone https://github.com/nramani611/scCont.git
cd scCont
pip install -e ".[dev]"
pytest
```

## Quickstart

scCont works on an [AnnData](https://anndata.readthedocs.io) object (cells x genes). Put any
per-cell labels you have (timepoint, condition, batch, cluster, ...) in `adata.obs`; they are
carried through every step and used for interpretation and plotting. Training itself is
unsupervised and never reads them. Results are written back onto the same object:

| Where | What |
|---|---|
| `adata.obsm["X_sccont"]` | latent features, cells x 32 |
| `adata.obsm["X_sccont_pca"]` | PCA of the latent space |
| `adata.varm["sccont_shap_mean_abs"]` | mean \|SHAP\| per gene and latent feature |
| `adata.uns["sccont"]` | latent key, latent dimension, spatial cluster labels |

```python
import sccont

sccont.set_seeds(42)                                   # all paper results use seed 42

# Any AnnData of raw counts works. The repo's genes x cells tables convert like this:
counts = sccont.load_expression_matrix("MCF10A_TGFB1/GSE200981_scRNAseq_processed.tsv")
adata = sccont.to_anndata(counts, timepoint=sccont.timepoints_from_columns(counts.columns))
# adata.obs["condition"] = ...                         # add any other labels you have

# Normalise (skip if adata.X is already normalised); obs labels are preserved
adata = sccont.preprocess(adata)                       # QC, Scrublet, log-norm, 3000 HVGs, regress, scale

# A) kNN positive pairs
pairs = sccont.get_knn_pairs(adata, k=3)

# B) Contrastive training + embedding (latents -> adata.obsm["X_sccont"])
encoder, projector, losses = sccont.train_contrastive(adata, pairs, latent_dim=32, epochs=20000)
sccont.embed(encoder, adata)
sccont.save_encoder(encoder, "encoder.pth")

# C) Spatial clustering of latent features
sccont.elbow_plot_for_clusters(adata, max_k=12)        # pick n_groups from the elbow
groups, _ = sccont.group_spatially_similar_latents(adata, n_groups=3)

# D) Network attribution (SHAP): cells x genes x latents; summary -> adata.varm
shap_values = sccont.compute_shap_values(encoder, adata)
gene_names = list(adata.var_names)
top_genes = sccont.select_top_genes_by_zscore(shap_values, gene_names, latent_feature_index=7)

# E) GO enrichment per spatial cluster (network call to g:Profiler)
go_results = sccont.enrich_latent_clusters(shap_values, gene_names, groups)
sccont.write_go_results_excel(go_results, "GO_Enrichment_Results.xlsx")

# Latent features as an AnnData of their own (obs labels copied) for plotting / scanpy tools
adata_latent = sccont.latent_anndata(adata)
```

Reusing a trained encoder shipped in this repository:

```python
encoder = sccont.load_encoder("MCF7_TNF/encoder.pth", input_dim=adata.n_vars, latent_dim=32)
```

Arrays still work everywhere an `AnnData` is accepted (cells x genes), and the original
`TemporalSingleCellDataset(genes_x_cells, timepoints)` container is kept for backwards compatibility.

Every preprocessing step has its own switch, so you can turn off the ones that do not apply to
your data. For example, simulated data with no mitochondrial or cell-cycle signal:

```python
adata = sccont.preprocess(
    adata,
    remove_doublets=False,      # skip Scrublet
    max_pct_mito=None,          # no mitochondrial filter
    regress_mito=False,
    regress_cell_cycle=False,
)
```

The full list of switches (`min_genes`, `max_genes`, `max_pct_mito`, `min_cells`, `remove_doublets`,
`normalize_total`, `log1p`, `n_top_genes`, `regress_mito`, `regress_counts`, `regress_cell_cycle`,
`scale`) is documented in `help(sccont.preprocess)`.

## API overview

| Stage | Function | Purpose |
|---|---|---|
| — | `set_seeds`, `get_device` | Reproducibility, device selection |
| — | `load_expression_matrix`, `to_anndata`, `timepoints_from_columns` | Read the repo's genes x cells tables and build an `AnnData` with labels in `obs` |
| — | `preprocess` | QC / normalise an `AnnData` (every step switchable), `obs` preserved |
| A | `get_knn_pairs` | kNN positive pairs in PCA space (labels ignored) |
| B | `Encoder`, `Projector`, `InfoNCELoss` | Network components |
| B | `train_contrastive`, `embed`, `latent_anndata`, `save_encoder`, `load_encoder` | Train; write latents to `adata.obsm`; latent-features-as-variables view; persist weights |
| C | `elbow_plot_for_clusters`, `group_spatially_similar_latents`, `latents_in_group` | Cluster latent features by spatial pattern (stored in `adata.uns`) |
| D | `compute_shap_values`, `select_top_genes_by_zscore`, `top_genes_per_latent` | Gene -> latent attributions (summary in `adata.varm`) |
| E | `enrich_latent_clusters`, `run_go_enrichment`, `write_go_results_excel`, `collect_enriched_genes` | GO enrichment |

Every function has a docstring; `help(sccont.preprocess)` lists all QC parameters.

## Tutorial notebook

`scCont Pipeline.ipynb` walks through the five stages on the MCF10A TGFB1 dataset using the
package API, then reproduces the latent-level analysis and manuscript figures (functional-group
heatmaps and multi-latent trajectory plots). Run it with the `cont-learn` kernel or any environment
where `sccont` and `jupyter` are installed.

## Simulated data (SERGIO)

The simulation benchmark in the manuscript uses [SERGIO](https://github.com/PayamDiba/SERGIO)
with explicit ground-truth gene programs. `simulation/` contains the exact scripts and per-run
parameter files used to generate every simulated dataset, plus a normalisation step that matches
the real-data pipeline. See [`simulation/README.md`](simulation/README.md) for setup and the
command that reproduces each dataset.

## Repository Structure & Data

There are 13 folders in this repository - one for each dataset benchmarked in the study. Due to GitHub file size limits, some of the `.gz` datasets are hosted externally. All full datasets used to train scCont are available via this [Google Drive link](https://drive.google.com/drive/folders/1sqRm1o5t8Tizw4sQfJXdQ-blWzii3jVh?usp=sharing).

The following files are available for all datasets:
* `.gz` file: The processed dataset used to train scCont.
* `GO_Enrichment_Results.xlsx`: GO enrichment results from each cluster (calculated post-Stage 3). Each sheet corresponds to a different spatial cluster.
* `encoder.pth`: The saved PyTorch weights of the encoder function after training (load with `sccont.load_encoder`).

Specific Dataset Annotations
The `MCF10A_TGFB1` folder and all `A549` dataset folders contain additional metadata used to generate manuscript figures:
* `functional_groups.json`: Dictionary mapping functional group names to the specific genes involved.
* `group_to_latent.json`: Dictionary mapping functional group names to their source latent feature index.
* `invert_shap.json`: Dictionary mapping latent features to a boolean indicating if SHAP values were inverted to establish a positive correlation with experimental time.
* `latent_to_bifurcation.json`: Dictionary mapping latent features to a boolean indicating if it was identified as a bifurcating trajectory _(Note: This analysis is specific to the MCF10A_TGFB1 dataset)._

Other files:
* `src/sccont/` – the installable package.
* `tests/` – pytest suite (runs on CPU in a few seconds).
* `requirements.txt` – the exact frozen environment used for the manuscript, kept for reproducibility. The package's own dependencies are declared in `pyproject.toml`.
* `RELEASING.md` – how releases are published to PyPI.

## Contact
If you have any questions, encounter issues, or need help adapting `scCont` for your own data, please open an issue on this repository or contact the corresponding author, Neal Kewalramani, at nramani@bu.edu.
