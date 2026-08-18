"""Error-analysis workbench: the *look at your failures first* core.

Workflow: sample failing traces -> embed them -> cluster into a failure taxonomy. You read
the clusters, name the failure modes, and codify each recurring one as a CodeCheck or a
judge rubric. Clustering is fully implemented (sklearn KMeans); the embedding encoder is a
pluggable seam left as a ``# TODO`` so you can choose on-device vs. hosted embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import KMeans

from .models import FailureCluster, Trace

# Pluggable embedding encoder. Left unset on purpose: pick your own cost/latency trade-off
# (e.g. a local sentence-transformers model, or a hosted embeddings endpoint).
# TODO: assign a callable(list[str]) -> np.ndarray of shape (len(texts), dim).
_ENCODER = None


def embed(texts: Sequence[str]) -> np.ndarray:
    """Embed texts into a 2-D float array (n_texts, dim).

    Raises until an encoder is configured — EvalGate refuses to silently fabricate vectors.
    """
    if _ENCODER is None:
        raise RuntimeError(
            "No embedding encoder configured. Set evalgate.analysis._ENCODER to a callable "
            "(list[str]) -> np.ndarray, e.g. a sentence-transformers model or an embeddings API."
        )
    vectors = np.asarray(_ENCODER(list(texts)), dtype=float)  # type: ignore[operator]
    if vectors.ndim != 2:
        raise ValueError(f"encoder must return a 2-D array, got shape {vectors.shape}")
    return vectors


def sample_failures(traces: Sequence[Trace], n: int, seed: int = 0) -> list[Trace]:
    """Uniformly sample up to ``n`` failing traces for manual + automated error analysis."""
    failures = [t for t in traces if t.is_failure]
    if n >= len(failures):
        return failures
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(failures), size=n, replace=False)
    return [failures[i] for i in idx]


def cluster_failures(
    embeddings: np.ndarray,
    k: int,
    texts: Sequence[str] | None = None,
    exemplars_per_cluster: int = 3,
    seed: int = 0,
) -> list[FailureCluster]:
    """Cluster failure embeddings into a taxonomy with KMeans.

    Args:
        embeddings: (n, dim) array of failure embeddings (see :func:`embed`).
        k: desired number of clusters (clamped to ``[1, n]``).
        texts: optional parallel texts; when given, exemplars are the raw texts closest to
            each centroid, otherwise placeholder ``trace[i]`` handles.
        exemplars_per_cluster: how many representative members to surface per cluster.

    Returns clusters sorted largest-first, so the taxonomy leads with your biggest failure mode.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.size == 0:
        return []
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (n, dim), got shape {embeddings.shape}")

    n = embeddings.shape[0]
    if texts is not None and len(texts) != n:
        raise ValueError(f"texts length {len(texts)} != n embeddings {n}")

    k = max(1, min(k, n))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(embeddings)

    clusters: list[FailureCluster] = []
    for c in range(k):
        member_idx = np.where(labels == c)[0]
        if member_idx.size == 0:
            continue
        centroid = km.cluster_centers_[c]
        dist = np.linalg.norm(embeddings[member_idx] - centroid, axis=1)
        nearest = member_idx[np.argsort(dist)][:exemplars_per_cluster]
        if texts is not None:
            exemplars = [texts[i] for i in nearest]
        else:
            exemplars = [f"trace[{i}]" for i in nearest]
        clusters.append(
            FailureCluster(label=f"cluster-{c}", size=int(member_idx.size), exemplars=exemplars)
        )

    clusters.sort(key=lambda fc: fc.size, reverse=True)
    return clusters
