"""Tests for sccont.annotate: reports (no LLM) and mocked Claude annotation."""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sccont
from sccont.annotate import FunctionalGroup, LatentAnnotation

pytest.importorskip("shap")
LATENT_DIM = 6


@pytest.fixture(scope="module")
def fitted(toy_matrix):
    obs = pd.DataFrame(
        {"timepoint": [c.split("_")[1] for c in toy_matrix.columns],
         "pseudotime": np.linspace(0, 1, toy_matrix.shape[1])},
        index=toy_matrix.columns,
    )
    adata = sccont.to_anndata(toy_matrix, obs=obs)
    pairs = sccont.get_knn_pairs(adata, k=3, n_pcs=20, random_state=0)
    encoder, _, _ = sccont.train_contrastive(adata, pairs, latent_dim=LATENT_DIM, proj_dim=4, epochs=30,
                                             batch_size=64, device="cpu", seed=42, verbose=False)
    sccont.embed(encoder, adata, device="cpu")
    groups, _ = sccont.group_spatially_similar_latents(adata, n_groups=2, bins=10)
    sv = sccont.compute_shap_values(encoder, adata, n_background=5, device="cpu")
    go = [pd.DataFrame({"name": ["term A", "term B"], "source": ["GO:BP", "GO:MF"],
                        "p_value": [1e-5, 1e-3], "intersection_size": [4, 3]}),
          pd.DataFrame()]
    return adata, sv, groups, go


def test_gene_directions_sign(fitted):
    adata, sv, _, _ = fitted
    sv2 = sv.copy()
    X = adata.X
    sv2[:, 0, 0] = X[:, 0] * 2.0        # positively correlated
    sv2[:, 1, 0] = -X[:, 1] * 2.0       # negatively correlated
    df = sccont.gene_directions(adata, sv2, 0, [adata.var_names[0], adata.var_names[1], "NOPE"])
    assert df["Direction"].tolist() == ["positive", "negative"]
    flipped = sccont.gene_directions(adata, sv2, 0, [adata.var_names[0]], invert=True)
    assert flipped["Direction"].tolist() == ["negative"]


def test_latent_report_contents(fitted):
    adata, sv, groups, go = fitted
    rep = sccont.latent_report(adata, sv, 0, go_results=go, groupby=["timepoint", "pseudotime"], zscore_threshold=0.5)
    assert rep.latent == 0 and rep.group == int(groups[0])
    assert len(rep.drivers) >= 2
    assert set(rep.genes) <= set(adata.var_names)
    assert set(rep.positive) | set(rep.negative) == set(rep.genes)
    txt = rep.to_prompt()
    assert "POSITIVE DRIVERS" in txt and "NEGATIVE DRIVERS" in txt
    assert rep.genes[0] in txt
    assert "timepoint" in txt and "pseudotime" in txt
    if rep.group == 0:
        assert "term A" in txt
    assert rep.to_frame().shape[0] == len(rep.drivers)


def test_latent_report_fallback_and_invert(fitted):
    adata, sv, _, _ = fitted
    rep = sccont.latent_report(adata, sv, 1, zscore_threshold=50.0, fallback_n_genes=7)
    assert len(rep.drivers) == 7 and rep.notes
    rep_inv = sccont.latent_report(adata, sv, 1, zscore_threshold=50.0, fallback_n_genes=7, invert=True)
    assert rep_inv.inverted
    assert {d.gene: d.direction for d in rep_inv.drivers} == {
        d.gene: ("negative" if d.direction == "positive" else "positive") for d in rep.drivers}


class _FakeMessages:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        ann = self.payloads.pop(0)
        return SimpleNamespace(parsed_output=ann, stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=120, output_tokens=40))


class _FakeClient:
    def __init__(self, payloads):
        self.messages = _FakeMessages(payloads)


def _fake_annotation(rep, extra_gene="NOT_A_GENE"):
    g = rep.genes
    return LatentAnnotation(latent=999, summary="test signal", groups=[
        FunctionalGroup(name="Group one", genes=[g[0], g[1], extra_gene, g[0].lower()], direction="positive",
                        rationale="r1", confidence="high"),
        FunctionalGroup(name="Too small", genes=[g[2]], direction="negative", rationale="r2", confidence="low"),
        FunctionalGroup(name="Group one", genes=[g[3], g[4]] if len(g) > 4 else [g[2], g[3]], direction="mixed",
                        rationale="r3", confidence="medium"),
    ])


def test_annotate_latent_cleans_output(fitted):
    adata, sv, _, go = fitted
    rep = sccont.latent_report(adata, sv, 0, go_results=go, groupby="timepoint", zscore_threshold=0.5)
    client = _FakeClient([_fake_annotation(rep)])
    ann, usage = sccont.annotate_latent(rep, client=client, model="claude-opus-5", max_groups=6, min_genes=2,
                                        extra_context="MCF7 cells, TNF time course", return_usage=True)
    assert ann.latent == 0
    names = [g.name for g in ann.groups]
    assert names == ["Group one", "Group one (2)"]           # duplicate renamed, small group dropped
    assert ann.groups[0].genes == [rep.genes[0], rep.genes[1]]  # hallucinated + duplicate removed
    assert usage == {"input_tokens": 120, "output_tokens": 40}
    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["output_format"] is LatentAnnotation
    assert call["thinking"] == {"type": "adaptive"}
    assert "TNF time course" in call["messages"][0]["content"]
    assert "at most 6" in call["system"]


def test_annotate_latent_refusal(fitted):
    adata, sv, _, _ = fitted
    rep = sccont.latent_report(adata, sv, 0, zscore_threshold=0.5)

    class Refusing:
        class messages:
            @staticmethod
            def parse(**kw):
                return SimpleNamespace(parsed_output=None, stop_reason="refusal", stop_details="x", usage=None)

    with pytest.raises(RuntimeError):
        sccont.annotate_latent(rep, client=Refusing())


def test_annotate_latents_and_save(fitted, tmp_path):
    adata, sv, _, go = fitted
    reps = [sccont.latent_report(adata, sv, i, zscore_threshold=0.5) for i in (0, 1)]
    client = _FakeClient([_fake_annotation(reps[0]), _fake_annotation(reps[1])])
    result = sccont.annotate_latents(adata, sv, [0, 1], go_results=go, groupby="timepoint", invert_shap={1: True},
                                     client=client, zscore_threshold=0.5, verbose=False)
    assert set(result.group_to_latent.values()) == {0, 1}
    assert len(result.functional_groups) == 4
    assert "Group one (LF 1)" in result.functional_groups          # unique across latents
    for name, genes in result.functional_groups.items():
        assert set(genes) <= set(adata.var_names)
    assert result.usage == {"input_tokens": 240, "output_tokens": 80}
    assert list(result.table.columns)[:3] == ["group", "latent", "n_genes"]
    assert adata.uns["sccont"]["functional_groups"] == result.functional_groups

    out = result.save(tmp_path)
    fg = json.loads((out / "functional_groups.json").read_text())
    g2l = json.loads((out / "group_to_latent.json").read_text())
    inv = {int(k): v for k, v in json.loads((out / "invert_shap.json").read_text()).items()}
    assert fg == result.functional_groups and g2l == result.group_to_latent and inv == {1: 1}
    assert (out / "annotations.csv").exists()
    # the saved files drive the plotting heatmap directly
    ax = sccont.pl.functional_group_heatmap(adata, sv, fg, g2l, groupby="timepoint", invert_shap=inv, show=False)
    assert ax is not None


def test_missing_anthropic_gives_install_hint(fitted, monkeypatch):
    adata, sv, _, _ = fitted
    rep = sccont.latent_report(adata, sv, 0, zscore_threshold=0.5)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ImportError, match="sccont\\[llm\\]"):
        sccont.annotate_latent(rep)


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="live Claude call; set ANTHROPIC_API_KEY")
def test_live_annotate(fitted):
    adata, sv, _, go = fitted
    rep = sccont.latent_report(adata, sv, 0, go_results=go, groupby="timepoint", zscore_threshold=0.5)
    ann = sccont.annotate_latent(rep, effort="low")
    assert all(set(g.genes) <= set(rep.genes) for g in ann.groups)
