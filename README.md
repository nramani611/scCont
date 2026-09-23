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
pip install sccont            # core pipeline (stages A-D)
pip install "sccont[all]"     # + Scrublet doublet removal and g:Profiler GO enrichment
```

Requires Python 3.10+. `pip` installs a CPU build of PyTorch by default; for GPU training install
torch first following <https://pytorch.org/get-started/locally/>, then install `sccont`.

To work from a source checkout instead:

```bash
git clone https://github.com/nramani611/scCont.git
cd scCont
pip install -e ".[all,dev]"
pytest
```

## Quickstart

Input: a genes x cells count table whose cell names encode the timepoint as `V<cell>_T<time>`
(e.g. `V1_T0`). The example below reproduces the notebook pipeline on the MCF10A TGFB1 dataset.

```python
import sccont

sccont.set_seeds(42)                                   # all paper results use seed 42

# Load and normalise (skip preprocess() if your matrix is already normalised)
counts = sccont.load_expression_matrix("MCF10A_TGFB1/GSE200981_scRNAseq_processed.tsv")
expr = sccont.preprocess(counts)                       # QC, Scrublet, log-norm, 3000 HVGs, regress, scale
timepoints = sccont.timepoints_from_columns(expr.columns)
dataset = sccont.TemporalSingleCellDataset(expr.to_numpy(), timepoints)

# A) kNN positive pairs
pairs = sccont.get_knn_pairs(dataset, k=3)

# B) Contrastive training + embedding
encoder, projector, losses = sccont.train_contrastive(dataset, pairs, latent_dim=32, epochs=20000)
adata_latent = sccont.embed(encoder, dataset)          # AnnData: cells x 32 latent features, PCA computed
sccont.save_encoder(encoder, "encoder.pth")

# C) Spatial clustering of latent features
sccont.elbow_plot_for_clusters(adata_latent, max_k=12) # pick n_groups from the elbow
groups, _ = sccont.group_spatially_similar_latents(adata_latent, n_groups=3)

# D) Network attribution (SHAP): cells x genes x latents
shap_values = sccont.compute_shap_values(encoder, expr.to_numpy().T)
gene_names = list(expr.index)
top_genes = sccont.select_top_genes_by_zscore(shap_values, gene_names, latent_feature_index=7)

# E) GO enrichment per spatial cluster (network call to g:Profiler)
go_results = sccont.enrich_latent_clusters(shap_values, gene_names, groups)
sccont.write_go_results_excel(go_results, "GO_Enrichment_Results.xlsx")
```

Reusing a trained encoder shipped in this repository:

```python
encoder = sccont.load_encoder("MCF7_TNF/encoder.pth", input_dim=expr.shape[0], latent_dim=32)
```

For simulated data with no mitochondrial or cell-cycle signal, call
`preprocess(counts, remove_doublets=False, regress_mito=False, regress_cell_cycle=False, max_pct_mito=None)`.

## API overview

| Stage | Function | Purpose |
|---|---|---|
| — | `set_seeds`, `get_device` | Reproducibility, device selection |
| — | `load_expression_matrix`, `preprocess`, `timepoints_from_columns` | Load genes x cells table; QC/normalise; parse `V*_T*` labels |
| — | `TemporalSingleCellDataset` | Holds the expression tensor and timepoints |
| A | `get_knn_pairs` | kNN positive pairs in PCA space |
| B | `Encoder`, `Projector`, `InfoNCELoss` | Network components |
| B | `train_contrastive`, `embed`, `save_encoder`, `load_encoder` | Train, embed to `AnnData`, persist weights |
| C | `elbow_plot_for_clusters`, `group_spatially_similar_latents`, `latents_in_group` | Cluster latent features by spatial pattern |
| D | `compute_shap_values`, `select_top_genes_by_zscore`, `top_genes_per_latent` | Gene -> latent attributions |
| E | `enrich_latent_clusters`, `run_go_enrichment`, `write_go_results_excel`, `collect_enriched_genes` | GO enrichment |

Every function has a docstring; `help(sccont.preprocess)` lists all QC parameters.

## Tutorial notebook

`scCont Pipeline.ipynb` walks through the five stages on the MCF10A TGFB1 dataset using the
package API, then reproduces the latent-level analysis and manuscript figures (functional-group
heatmaps and multi-latent trajectory plots). Run it with the `cont-learn` kernel or any environment
where `sccont[all]` and `jupyter` are installed.

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
