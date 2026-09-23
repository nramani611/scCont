"""scCont: unsupervised contrastive learning of interpretable latent features from scRNA-seq.

The pipeline works on an ``AnnData`` (cells x genes). Any per-cell labels live in
``adata.obs`` and are carried through untouched; training is unsupervised.

A. :func:`get_knn_pairs`                    – kNN positive-pair selection
B. :func:`train_contrastive`, :func:`embed` – InfoNCE training; latents -> ``adata.obsm['X_sccont']``
C. :func:`group_spatially_similar_latents`  – spatial clustering of latent features
D. :func:`compute_shap_values`              – network attribution (SHAP); summary -> ``adata.varm``
E. :func:`enrich_latent_clusters`           – GO enrichment of latent clusters
E. :func:`latent_report`, :func:`annotate_latents` – per-latent driver report; optional Claude-assisted functional groups
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sccont")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

from .annotate import (
    AnnotationResult,
    LatentReport,
    annotate_latent,
    annotate_latents,
    gene_directions,
    latent_report,
)
from .annotation import (
    cluster_gene_sets,
    collect_enriched_genes,
    enrich_latent_clusters,
    run_go_enrichment,
    write_go_results_excel,
)
from .attribution import (
    SHAP_VARM_KEY,
    compute_shap_values,
    select_top_genes_by_zscore,
    top_genes_per_latent,
)
from .data import SingleCellDataset, TemporalSingleCellDataset, as_dataset
from .losses import InfoNCELoss
from .models import Encoder, Projector
from .pairs import get_knn_pairs, get_knn_pairs_agnostic
from .preprocessing import (
    G2M_GENES,
    S_GENES,
    load_expression_matrix,
    preprocess,
    timepoints_from_columns,
    to_anndata,
)
from .spatial import (
    elbow_plot_for_clusters,
    group_spatially_similar_latents,
    latents_in_group,
    spatial_map,
)
from .training import (
    LATENT_KEY,
    embed,
    latent_anndata,
    load_encoder,
    save_encoder,
    train_contrastive,
)
from .utils import get_device, set_seeds
from . import plotting as pl

__all__ = [
    "__version__",
    "pl",
    # utils
    "set_seeds",
    "get_device",
    # preprocessing / input
    "load_expression_matrix",
    "to_anndata",
    "preprocess",
    "timepoints_from_columns",
    "S_GENES",
    "G2M_GENES",
    # data / pairs
    "SingleCellDataset",
    "TemporalSingleCellDataset",
    "as_dataset",
    "get_knn_pairs",
    "get_knn_pairs_agnostic",
    # models / training
    "Encoder",
    "Projector",
    "InfoNCELoss",
    "train_contrastive",
    "embed",
    "latent_anndata",
    "LATENT_KEY",
    "save_encoder",
    "load_encoder",
    # spatial
    "spatial_map",
    "group_spatially_similar_latents",
    "elbow_plot_for_clusters",
    "latents_in_group",
    # attribution
    "compute_shap_values",
    "SHAP_VARM_KEY",
    "select_top_genes_by_zscore",
    "top_genes_per_latent",
    # annotation (cluster level)
    "run_go_enrichment",
    "cluster_gene_sets",
    "enrich_latent_clusters",
    "write_go_results_excel",
    "collect_enriched_genes",
    # annotation (latent level, optional LLM)
    "gene_directions",
    "latent_report",
    "annotate_latent",
    "annotate_latents",
    "LatentReport",
    "AnnotationResult",
]
