# scCont
`scCont` is an unsupervised contrastive deep learning model that extracts interpretable latent features from single-cell RNA data.

Contrastive learning is typically used to find similarities between actual and perturbed data, but we modify the typical contrastive learning framework to use the framework to learn local neighborhood functional group information in scRNA data. scCont achieves this through five steps: 1) _k_NN Pair Selection, 2) Contrastive Training, 3) Signal Identification via Spatial Clustering, 4) Network Attribution Analysis, 5) Biological Annoatation. Figure 1 visualizes these steps.


![scCont Pipeline](Figure1_V2.png)
The framework operates in five stages to extract interpretable latent features (LFs) from timecourse scRNA-seq data. **A _k_-NN Pair Selection**: A _k_-Nearest Neighbor graph (K=3) is applied to the normalized gene expression space to identify positive pairs, preserving local topology. PCA is used here to reduce noise in the data. **B) Contrastive Training:** These pairs serve as inputs to a contrastive learning architecture that minimizes InfoNCE loss to maximize agreement between neighboring cells in the latent space. **C) Signal Identification via Spatial Clustering:** LF activations are projected onto the gene expression PCA space; features with correlated spatial patterns are grouped to isolate distinct signal trajectories. **D) Network Attribution Analysis:** Specific genes driving each LF are identified using feature importance scores (SHAP). **E) Biological Annotations:** LFs are connected to biological processes via a multi-tiered validation approach, combining broad Gene Ontology (GO) enrichment for clusters with functional group identification for individual top-contributing genes.

## Getting Started

### Prerequisites
To run `scCont`, clone this repository, and install the required dependencies. We highly recommend making a new python virtual environment for installing dependencies:

```bash
git clone [https://github.com/nramani611/scCont.git](https://github.com/YOUR_USERNAME/scCont.git)
cd scCont
pip install -r requirements.txt
```

## Overview of the Repository 

The `scCont` pipeline jupyter notebook in this repository is designed to walk through the code for each of the five steps and offers guidance on how to apply this pipeline to the reader's own scRNA data. The current cells in the notebook outline the code used to generate the figures seen for Figures 2 and 3 in the paper. There are 13 folders in this repository - one for each dataset with the following files:

These files are in the MCF10A_TGFB1 and all the A549 folders. These were generated and used to make the figures seen in Figures 3 and 4:
* functional_groups.json: Dictionary mapping functional group names to genes involved
* group_to_latent.json: Dictionary mapping functional group name to the latent feature number it came from
* invert_shap.json: Dictionary mapping latent to boolean to see if latent values need to be inverted to properly make time and latent features positively correlated (this is purely done to make it so all latent features are positively correleted with time)
* latent_to_bifurcation.json: Dictionary mapping latent to boolean to see if it was identified as a bifurcating latent (this was only done for the MCF10A_TGFB1 dataset)

The following files are available for all datasets:
* .gz file: Dataset used to train scCont (note some of the files were too big for GitHub - all datasets used are available on the [Google Drive link](https://drive.google.com/drive/folders/1sqRm1o5t8Tizw4sQfJXdQ-blWzii3jVh?usp=sharing))
* GO_Enrichment_Results.xlsx: GO enrichment results from each cluster. Enrichment was done after Figure 1C. Each sheet corresponds to a different cluster
* encoder.pth: Saved weights of the encoder function after training

If you have any questions, please feel free to create an issue on this repository or email the correpsonding author, Neal Kewalramani, at nramani@bu.edu.
