"""Stage E (latent level): per-latent reports and LLM-assisted functional-group annotation.

Two layers:

* :func:`latent_report` needs no LLM. It gathers, for one latent feature, the
  z-score top genes labelled as positive or negative drivers, the GO terms
  enriched in the latent's spatial cluster, and which ``obs`` label categories
  the latent is high or low in. ``LatentReport.to_prompt()`` renders the text
  that can be pasted into any assistant (Claude, ChatGPT, ...).
* :func:`annotate_latent` / :func:`annotate_latents` send that report to Claude
  through the official ``anthropic`` SDK (``pip install "sccont[llm]"``) and
  return functional groups in the same format as the repository's
  ``functional_groups.json`` / ``group_to_latent.json`` files.

The LLM output is a *hypothesis* for expert review: every group carries a
rationale and a confidence, and genes not present in the report are discarded.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .attribution import select_top_genes_by_zscore
from .training import LATENT_KEY

DEFAULT_MODEL = "claude-opus-5"
INSTALL_HINT = 'LLM annotation needs the anthropic SDK: pip install "sccont[llm]"'

try:  # pydantic ships with the llm extra (and with most scanpy stacks)
    from pydantic import BaseModel, Field

    class FunctionalGroup(BaseModel):
        """One proposed functional group / pathway among a latent's driver genes."""

        name: str = Field(description="Short, literature-grounded pathway or process name")
        genes: list[str] = Field(description="Genes from the provided driver list only")
        direction: Literal["positive", "negative", "mixed"] = Field(
            description="positive = high expression increases the latent feature; negative = suppresses"
        )
        rationale: str = Field(description="One or two sentences on why these genes belong together")
        confidence: Literal["high", "medium", "low"]

    class LatentAnnotation(BaseModel):
        """Structured LLM answer for one latent feature."""

        latent: int
        summary: str = Field(description="One sentence: what biological signal this latent feature most likely captures")
        groups: list[FunctionalGroup]

    _HAVE_PYDANTIC = True
except ImportError:  # pragma: no cover
    FunctionalGroup = LatentAnnotation = None  # type: ignore[assignment]
    _HAVE_PYDANTIC = False


# --------------------------------------------------------------------------- #
# Report (no LLM)
# --------------------------------------------------------------------------- #
def _dense(X) -> np.ndarray:
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _latent_matrix(adata: ad.AnnData, use_rep: str | None = None) -> np.ndarray | None:
    key = use_rep or adata.uns.get("sccont", {}).get("latent_key", LATENT_KEY)
    if key in adata.obsm:
        return np.asarray(adata.obsm[key])
    return None


def gene_directions(
    adata: ad.AnnData,
    shap_values: np.ndarray,
    latent: int,
    genes: Sequence[str],
    invert: bool = False,
) -> pd.DataFrame:
    """Whether high expression of each gene increases or suppresses a latent feature.

    The sign of the Pearson correlation between a gene's expression and its
    SHAP value for ``latent`` gives the direction (port of the notebook's
    ``interpret_shap_directions``). ``invert=True`` flips the sign, used when a
    latent is anti-correlated with the label of interest so that "positive"
    still means "promotes the process".

    Returns columns ``Gene``, ``Correlation_to_SHAP``, ``Direction``
    (``'positive'``/``'negative'``) and ``Interpretation``.
    """
    X = _dense(adata.X)
    pos = {g: i for i, g in enumerate(adata.var_names)}
    rows = []
    for g in genes:
        if g not in pos:
            continue
        j = pos[g]
        x, s = X[:, j], shap_values[:, j, latent]
        corr = float(pearsonr(x, s)[0]) if np.std(x) > 0 and np.std(s) > 0 else 0.0
        positive = (corr > 0) != bool(invert)
        rows.append(
            {
                "Gene": g,
                "Correlation_to_SHAP": round(corr, 3),
                "Direction": "positive" if positive else "negative",
                "Interpretation": "High expression promotes the process" if positive
                else "High expression suppresses the process",
            }
        )
    return pd.DataFrame(rows, columns=["Gene", "Correlation_to_SHAP", "Direction", "Interpretation"])


@dataclass
class GeneDriver:
    gene: str
    mean_abs_shap: float
    z_score: float
    correlation: float
    direction: str  # "positive" | "negative"


@dataclass
class LatentReport:
    """Everything known about one latent feature, ready to show a human or an LLM."""

    latent: int
    drivers: list[GeneDriver]
    group: int | None = None
    go_terms: list[dict] = field(default_factory=list)
    label_summary: dict[str, dict] = field(default_factory=dict)
    inverted: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def positive(self) -> list[str]:
        return [d.gene for d in self.drivers if d.direction == "positive"]

    @property
    def negative(self) -> list[str]:
        return [d.gene for d in self.drivers if d.direction == "negative"]

    @property
    def genes(self) -> list[str]:
        return [d.gene for d in self.drivers]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([d.__dict__ for d in self.drivers])

    def to_prompt(self) -> str:
        """Plain-text description to paste into an assistant or send via :func:`annotate_latent`."""
        lines = [f"Latent feature {self.latent} from an scCont contrastive model of scRNA-seq data."]
        if self.group is not None:
            lines.append(f"It belongs to spatial cluster {self.group} of latent features.")
        if self.inverted:
            lines.append("The sign of this feature was flipped so that 'positive' means 'increases with the process of interest'.")
        lines.append("")
        lines.append("POSITIVE DRIVERS (high expression increases this feature), strongest first:")
        lines.append("  " + (", ".join(self.positive) if self.positive else "(none)"))
        lines.append("")
        lines.append("NEGATIVE DRIVERS (high expression suppresses this feature), strongest first:")
        lines.append("  " + (", ".join(self.negative) if self.negative else "(none)"))
        if self.label_summary:
            lines.append("")
            lines.append("LABEL ASSOCIATION of the feature's activation:")
            for col, summ in self.label_summary.items():
                if "correlation" in summ:
                    lines.append(f"  {col}: Pearson r = {summ['correlation']:+.2f}")
                else:
                    hi = ", ".join(map(str, summ.get("high", []))) or "-"
                    lo = ", ".join(map(str, summ.get("low", []))) or "-"
                    lines.append(f"  {col}: high in [{hi}]; low in [{lo}]")
        if self.go_terms:
            lines.append("")
            lines.append("GO TERMS enriched in this feature's spatial cluster (g:Profiler):")
            for t in self.go_terms:
                p = t.get("p_value")
                ptxt = f", p = {p:.1e}" if isinstance(p, (int, float)) else ""
                lines.append(f"  - {t.get('name')} ({t.get('source', 'GO')}{ptxt})")
        if self.notes:
            lines.append("")
            lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


def _label_summary(adata, latent_vec, groupby, invert: bool, z_cut: float = 0.5) -> dict:
    out = {}
    cols = [groupby] if isinstance(groupby, str) else list(groupby)
    v = -latent_vec if invert else latent_vec
    for col in cols:
        if col not in adata.obs:
            continue
        s = adata.obs[col]
        if pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
            y = s.to_numpy(dtype=float)
            ok = np.isfinite(y)
            r = float(pearsonr(v[ok], y[ok])[0]) if ok.sum() > 2 and np.std(y[ok]) > 0 else 0.0
            out[col] = {"correlation": r}
        else:
            means = pd.Series(v).groupby(s.to_numpy()).mean()
            if len(means) > 1 and means.std() > 0:
                z = (means - means.mean()) / means.std()
            else:
                z = means * 0
            out[col] = {
                "high": [k for k, val in z.sort_values(ascending=False).items() if val > z_cut],
                "low": [k for k, val in z.sort_values().items() if val < -z_cut],
                "mean_activation": {str(k): float(val) for k, val in means.items()},
            }
    return out


def latent_report(
    adata: ad.AnnData,
    shap_values: np.ndarray,
    latent: int,
    *,
    go_results: Sequence[pd.DataFrame] | None = None,
    groups=None,
    groupby: str | Sequence[str] | None = None,
    zscore_threshold: float = 2.57,
    fallback_n_genes: int = 30,
    n_go_terms: int = 15,
    invert: bool = False,
    use_rep: str | None = None,
) -> LatentReport:
    """Collect the driver genes, GO context and label association for one latent feature.

    Parameters
    ----------
    adata
        The ``AnnData`` used for SHAP (``var_names`` = genes, ``obs`` = labels).
    shap_values
        ``(n_cells, n_genes, latent_dim)`` from :func:`sccont.compute_shap_values`
        computed on ``adata``.
    latent
        Latent feature index.
    go_results
        Per-cluster GO tables from :func:`sccont.enrich_latent_clusters`; the
        table of this latent's spatial cluster contributes the top ``n_go_terms``.
    groups
        Spatial cluster per latent (default: ``adata.uns['sccont']['latent_groups']``).
    groupby
        One or more ``obs`` columns to summarise the feature against.
    zscore_threshold
        Passed to :func:`sccont.select_top_genes_by_zscore`; if it yields fewer
        than two genes the top ``fallback_n_genes`` by mean |SHAP| are used instead.
    invert
        Flip the sign of the feature (see :func:`gene_directions`).
    """
    latent = int(latent)
    gene_names = list(adata.var_names)
    top = select_top_genes_by_zscore(shap_values, gene_names, latent, zscore_threshold)
    notes = []
    if len(top) < 2:
        mean_abs = np.abs(shap_values[:, :, latent]).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:fallback_n_genes]
        from scipy.stats import zscore

        z = zscore(mean_abs)
        top = pd.DataFrame({"Gene": [gene_names[i] for i in idx], "Mean_SHAP_Value": mean_abs[idx], "Z_Score": z[idx]})
        notes.append(f"fewer than two genes passed z >= {zscore_threshold}; showing the top {len(top)} genes by mean |SHAP|")

    dirs = gene_directions(adata, shap_values, latent, top["Gene"].tolist(), invert=invert).set_index("Gene")
    drivers = [
        GeneDriver(
            gene=row.Gene,
            mean_abs_shap=float(row.Mean_SHAP_Value),
            z_score=float(row.Z_Score),
            correlation=float(dirs.loc[row.Gene, "Correlation_to_SHAP"]),
            direction=str(dirs.loc[row.Gene, "Direction"]),
        )
        for row in top.itertuples(index=False)
    ]

    if groups is None:
        groups = adata.uns.get("sccont", {}).get("latent_groups")
    group = int(np.asarray(groups)[latent]) if groups is not None else None

    go_terms: list[dict] = []
    if go_results is not None and group is not None and group < len(go_results):
        df = go_results[group]
        if df is not None and len(df) and "p_value" in df:
            df = df.sort_values("p_value").head(n_go_terms)
            for _, r in df.iterrows():
                go_terms.append({k: r[k] for k in ("name", "source", "p_value", "intersection_size") if k in r})

    label_summary = {}
    if groupby is not None:
        lat = _latent_matrix(adata, use_rep)
        if lat is None:
            notes.append("no latent representation on adata; label association skipped (run sccont.embed)")
        else:
            label_summary = _label_summary(adata, lat[:, latent], groupby, invert)

    return LatentReport(latent=latent, drivers=drivers, group=group, go_terms=go_terms,
                        label_summary=label_summary, inverted=invert, notes=notes)


# --------------------------------------------------------------------------- #
# LLM annotation
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are an expert in molecular cell biology and single-cell RNA-seq interpretation.

You will receive a description of one latent feature learned by scCont, an unsupervised contrastive
model of scRNA-seq data: the genes that drive it (split into positive and negative drivers by the sign of
their expression-to-SHAP correlation), which sample labels the feature is high or low in, and Gene Ontology
terms enriched in the feature's cluster.

Your task: organise the driver genes into coherent functional groups (pathways, processes, cell states or
complexes) with literature-grounded names, so a biologist can interpret the feature.

Rules:
- Use ONLY genes from the provided driver lists, spelled exactly as given. Never add genes.
- Keep positive and negative drivers in separate groups unless a pathway genuinely contains both
  (then mark it "mixed").
- Prefer a few well-supported groups (at most {max_groups}) over many speculative ones; a gene may be left
  ungrouped, and a gene may appear in at most one group.
- Give a one- or two-sentence rationale per group citing the biology, and a confidence rating.
- Name groups in the style of the source literature (e.g. "Interferon-stimulated genes", "Epithelial
  keratins", "Ribosome biogenesis"). Keep names under 40 characters.
- This is a hypothesis for expert review, not a final annotation; do not overstate certainty."""


def _get_client(client):
    if client is not None:
        return client
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(INSTALL_HINT) from exc
    return anthropic.Anthropic()


def _clean_annotation(ann: "LatentAnnotation", report: LatentReport, max_groups: int, min_genes: int) -> "LatentAnnotation":
    allowed = {g.upper(): g for g in report.genes}
    cleaned, seen_names, used_genes = [], set(), set()
    for grp in ann.groups:
        genes = []
        for g in grp.genes:
            canon = allowed.get(str(g).strip().upper())
            if canon and canon not in genes and canon not in used_genes:
                genes.append(canon)
        if len(genes) < min_genes:
            continue
        name = grp.name.strip()
        base, k = name, 2
        while name.lower() in seen_names:
            name = f"{base} ({k})"
            k += 1
        seen_names.add(name.lower())
        used_genes.update(genes)
        cleaned.append(grp.model_copy(update={"name": name, "genes": genes}))
        if len(cleaned) >= max_groups:
            break
    return ann.model_copy(update={"latent": report.latent, "groups": cleaned})


def annotate_latent(
    report: LatentReport,
    *,
    model: str = DEFAULT_MODEL,
    client=None,
    max_groups: int = 6,
    min_genes: int = 2,
    effort: str = "high",
    extra_context: str | None = None,
    max_tokens: int = 16000,
    return_usage: bool = False,
):
    """Ask Claude to organise a latent's driver genes into named functional groups.

    Parameters
    ----------
    report
        From :func:`latent_report`.
    model
        Claude model ID (default ``claude-opus-5``).
    client
        An ``anthropic.Anthropic`` (or compatible) client. ``None`` creates one from
        ``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile.
    max_groups, min_genes
        Keep at most ``max_groups`` groups with at least ``min_genes`` valid genes each.
    effort
        ``output_config.effort``: ``low`` | ``medium`` | ``high`` | ``xhigh`` | ``max``.
    extra_context
        Optional free text appended to the prompt (tissue, perturbation, hypotheses).
    return_usage
        Also return a dict with ``input_tokens`` / ``output_tokens``.

    Returns
    -------
    LatentAnnotation (and usage dict if ``return_usage``)
        Genes not in the report are dropped and groups below ``min_genes`` removed.
    """
    if not _HAVE_PYDANTIC:  # pragma: no cover
        raise ImportError(INSTALL_HINT)
    client = _get_client(client)
    user_text = report.to_prompt()
    if extra_context:
        user_text += "\n\nADDITIONAL CONTEXT FROM THE ANALYST:\n" + extra_context.strip()

    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        system=SYSTEM_PROMPT.format(max_groups=max_groups),
        messages=[{"role": "user", "content": user_text}],
        output_format=LatentAnnotation,
    )
    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        raise RuntimeError(f"Claude declined to annotate latent {report.latent}: {details}")
    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError(f"No structured output returned for latent {report.latent} (stop_reason={getattr(response, 'stop_reason', None)})")
    ann = _clean_annotation(parsed, report, max_groups, min_genes)
    usage = getattr(response, "usage", None)
    usage_d = {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }
    return (ann, usage_d) if return_usage else ann


@dataclass
class AnnotationResult:
    """Functional groups for several latents in the repository's JSON layout."""

    functional_groups: dict[str, list[str]]
    group_to_latent: dict[str, int]
    invert_shap: dict[int, bool]
    table: pd.DataFrame
    annotations: list
    reports: list[LatentReport]
    usage: dict[str, int]

    def save(self, directory: str | Path) -> Path:
        """Write ``functional_groups.json``, ``group_to_latent.json``, ``invert_shap.json`` and ``annotations.csv``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "functional_groups.json").write_text(json.dumps(self.functional_groups, indent=2))
        (directory / "group_to_latent.json").write_text(json.dumps(self.group_to_latent, indent=2))
        (directory / "invert_shap.json").write_text(json.dumps({str(k): int(v) for k, v in self.invert_shap.items()}, indent=2))
        self.table.to_csv(directory / "annotations.csv", index=False)
        return directory


def annotate_latents(
    adata: ad.AnnData,
    shap_values: np.ndarray,
    latents: Sequence[int],
    *,
    go_results: Sequence[pd.DataFrame] | None = None,
    groups=None,
    groupby: str | Sequence[str] | None = None,
    invert_shap: Mapping[int, bool] | None = None,
    model: str = DEFAULT_MODEL,
    client=None,
    zscore_threshold: float = 2.57,
    n_go_terms: int = 15,
    max_groups: int = 6,
    min_genes: int = 2,
    effort: str = "high",
    extra_context: str | None = None,
    verbose: bool = True,
) -> AnnotationResult:
    """Annotate several latent features and assemble ``functional_groups`` / ``group_to_latent``.

    Group names are made unique across latents by appending ``" (LF <i>)"`` when
    two latents propose the same name. Results are also stored in
    ``adata.uns['sccont']`` under ``functional_groups``, ``group_to_latent`` and
    ``annotation_table``.
    """
    invert_shap = {int(k): bool(v) for k, v in (invert_shap or {}).items()}
    functional_groups: dict[str, list[str]] = {}
    group_to_latent: dict[str, int] = {}
    rows, annotations, reports = [], [], []
    usage = {"input_tokens": 0, "output_tokens": 0}

    for latent in latents:
        latent = int(latent)
        rep = latent_report(adata, shap_values, latent, go_results=go_results, groups=groups, groupby=groupby,
                            zscore_threshold=zscore_threshold, n_go_terms=n_go_terms,
                            invert=invert_shap.get(latent, False))
        ann, u = annotate_latent(rep, model=model, client=client, max_groups=max_groups, min_genes=min_genes,
                                 effort=effort, extra_context=extra_context, return_usage=True)
        usage["input_tokens"] += u["input_tokens"]
        usage["output_tokens"] += u["output_tokens"]
        for grp in ann.groups:
            name = grp.name
            if name in functional_groups:
                name = f"{grp.name} (LF {latent})"
                k = 2
                while name in functional_groups:
                    name = f"{grp.name} (LF {latent}, {k})"
                    k += 1
            functional_groups[name] = list(grp.genes)
            group_to_latent[name] = latent
            rows.append({"group": name, "latent": latent, "n_genes": len(grp.genes), "genes": ";".join(grp.genes),
                         "direction": grp.direction, "confidence": grp.confidence, "rationale": grp.rationale,
                         "latent_summary": ann.summary})
        annotations.append(ann)
        reports.append(rep)
        if verbose:
            print(f"LF {latent}: {len(ann.groups)} groups – {ann.summary}")

    table = pd.DataFrame(rows, columns=["group", "latent", "n_genes", "genes", "direction", "confidence", "rationale", "latent_summary"])
    adata.uns["sccont"] = {
        **adata.uns.get("sccont", {}),
        "functional_groups": functional_groups,
        "group_to_latent": group_to_latent,
        "annotation_table": table.to_dict(orient="records"),
    }
    return AnnotationResult(functional_groups, group_to_latent, invert_shap, table, annotations, reports, usage)


__all__ = [
    "DEFAULT_MODEL", "GeneDriver", "LatentReport", "AnnotationResult",
    "gene_directions", "latent_report", "annotate_latent", "annotate_latents",
    "FunctionalGroup", "LatentAnnotation", "SYSTEM_PROMPT",
]
