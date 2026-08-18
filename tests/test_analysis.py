"""Tests for the error-analysis workbench (embedding + taxonomy clustering)."""

import numpy as np

from evalgate.analysis import TfidfEncoder, build_taxonomy, cluster_failures, embed


def test_tfidf_encoder_returns_2d():
    vecs = TfidfEncoder(dim=8)(["read the users table", "join orders and refunds",
                                "post a summary to slack", "filter transactions"])
    assert vecs.ndim == 2
    assert vecs.shape[0] == 4


def test_embed_uses_default_encoder():
    vecs = embed(["alpha beta", "gamma delta", "alpha beta gamma"])
    assert vecs.ndim == 2
    assert vecs.shape[0] == 3


def test_build_taxonomy_separates_distinct_groups():
    # Two clearly different failure families -> should cluster into (at least) two groups.
    texts = (
        ["missing final sink node output slack post"] * 6
        + ["hallucinated webhook tool type not in grammar"] * 6
    )
    clusters = build_taxonomy(texts, k=2, seed=0)
    assert len(clusters) == 2
    assert sum(c.size for c in clusters) == 12
    # Each cluster is auto-labeled with distinctive terms.
    assert all(c.terms for c in clusters)
    joined = " ".join(t for c in clusters for t in c.terms)
    assert "webhook" in joined or "sink" in joined


def test_build_taxonomy_empty_input():
    assert build_taxonomy([]) == []


def test_build_taxonomy_auto_k_runs():
    texts = [f"failure mode {i % 3} some words here token{i}" for i in range(15)]
    clusters = build_taxonomy(texts, k=None, seed=0)
    assert clusters
    assert sum(c.size for c in clusters) == 15


def test_cluster_failures_sizes_and_order():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, size=(10, 4)) + np.array([5, 5, 5, 5])
    b = rng.normal(0, 0.1, size=(4, 4)) + np.array([-5, -5, -5, -5])
    emb = np.vstack([a, b])
    clusters = cluster_failures(emb, k=2, seed=0)
    assert [c.size for c in clusters] == [10, 4]  # largest first
