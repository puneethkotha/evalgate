"""Error-analysis workbench: the *look at your failures first* core.

Workflow (Hamel Husain / Shreya Shankar): sample failing traces -> embed -> cluster into a
failure taxonomy. You read the clusters, name the failure modes, and codify each recurring one
as a CodeCheck or a judge rubric. This is the discipline, not a dashboard: you can't write good
evaluators for failure modes you haven't observed.

The default embedding encoder is a **$0, offline, deterministic TF-IDF + SVD** encoder (pure
scikit-learn — no model download, no API, works in airplane mode), which matches EvalGate's
cost-sensitivity. Swap it for sentence-transformers or a hosted embeddings endpoint via
:func:`set_encoder` when you want semantic embeddings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import FailureCluster

Encoder = Callable[[Sequence[str]], np.ndarray]


class TfidfEncoder:
    """Deterministic, dependency-light text encoder: TF-IDF -> dense via truncated SVD (LSA).

    No network, no model weights to download — good enough to cluster failing traces into a
    taxonomy, and $0 to run. ``dim`` is the target dense dimensionality (clamped to what the
    vocabulary/sample supports).
    """

    def __init__(self, dim: int = 64, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t or "" for t in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=float)
        tfidf = TfidfVectorizer(stop_words="english", min_df=1).fit_transform(texts)
        n_features = tfidf.shape[1]
        if n_features == 0:
            return np.zeros((len(texts), 1), dtype=float)
        # SVD needs n_components < n_features and < n_samples.
        components = max(1, min(self.dim, n_features - 1, len(texts) - 1)) or 1
        if components < 1 or n_features <= 1 or len(texts) <= 1:
            return np.asarray(tfidf.todense(), dtype=float)
        svd = TruncatedSVD(n_components=components, random_state=self.seed)
        return np.asarray(svd.fit_transform(tfidf), dtype=float)


_ENCODER: Encoder = TfidfEncoder()


def set_encoder(encoder: Encoder) -> None:
    """Install a custom embedding encoder: ``callable(list[str]) -> np.ndarray (n, dim)``."""
    global _ENCODER
    _ENCODER = encoder


def get_encoder() -> Encoder:
    return _ENCODER


def embed(texts: Sequence[str]) -> np.ndarray:
    """Embed texts into a 2-D float array (n_texts, dim) using the configured encoder."""
    vectors = np.asarray(_ENCODER(list(texts)), dtype=float)
    if vectors.ndim != 2:
        raise ValueError(f"encoder must return a 2-D array, got shape {vectors.shape}")
    return vectors


def sample_failures(traces, n: int, seed: int = 0):
    """Uniformly sample up to ``n`` failing traces for manual + automated error analysis."""
    failures = [t for t in traces if t.is_failure]
    if n >= len(failures):
        return failures
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(failures), size=n, replace=False)
    return [failures[i] for i in idx]


def _choose_k(n: int, k: int | None, embeddings: np.ndarray, seed: int) -> int:
    if k is not None:
        return max(1, min(k, n))
    if n < 4:
        return max(1, n)
    # Pick k in [2, 6] by silhouette; falls back to 2 if scoring is degenerate.
    from sklearn.metrics import silhouette_score

    best_k, best_score = 2, -1.0
    for cand in range(2, min(6, n - 1) + 1):
        labels = KMeans(n_clusters=cand, n_init=10, random_state=seed).fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(embeddings, labels)
        except ValueError:
            continue
        if score > best_score:
            best_k, best_score = cand, score
    return best_k


def cluster_failures(
    embeddings: np.ndarray,
    k: int,
    texts: Sequence[str] | None = None,
    exemplars_per_cluster: int = 3,
    seed: int = 0,
) -> list[FailureCluster]:
    """Cluster failure embeddings into a taxonomy with KMeans (largest cluster first)."""
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
        exemplars = [texts[i] for i in nearest] if texts is not None else [
            f"trace[{i}]" for i in nearest
        ]
        clusters.append(
            FailureCluster(label=f"cluster-{c}", size=int(member_idx.size), exemplars=exemplars)
        )

    clusters.sort(key=lambda fc: fc.size, reverse=True)
    return clusters


def build_taxonomy(
    texts: Sequence[str],
    k: int | None = None,
    exemplars_per_cluster: int = 3,
    top_terms: int = 4,
    seed: int = 0,
) -> list[FailureCluster]:
    """Cluster failing-trace texts into an auto-labeled failure taxonomy.

    Unlike :func:`cluster_failures`, this runs TF-IDF directly so it can name each cluster by
    its most distinctive terms (open-coding assist) — the label is a starting point you rename.
    ``k=None`` picks the cluster count by silhouette score.
    """
    texts = [t or "" for t in texts]
    n = len(texts)
    if n == 0:
        return []

    vec = TfidfVectorizer(stop_words="english", min_df=1)
    tfidf = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    dense = embed(texts)  # for clustering + silhouette

    k = _choose_k(n, k, dense, seed)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(dense)
    tfidf_dense = np.asarray(tfidf.todense())

    clusters: list[FailureCluster] = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        centroid = dense[idx].mean(axis=0)
        dist = np.linalg.norm(dense[idx] - centroid, axis=1)
        nearest = idx[np.argsort(dist)][:exemplars_per_cluster]
        exemplars = [texts[i] for i in nearest]
        # Distinctive terms = highest mean TF-IDF within the cluster.
        terms: list[str] = []
        if vocab.size:
            mean_tfidf = tfidf_dense[idx].mean(axis=0)
            order = np.argsort(mean_tfidf)[::-1]
            terms = [vocab[j] for j in order[:top_terms] if mean_tfidf[j] > 0]
        label = ", ".join(terms) if terms else f"cluster-{c}"
        clusters.append(
            FailureCluster(label=label, size=int(idx.size), exemplars=exemplars, terms=terms)
        )

    clusters.sort(key=lambda fc: fc.size, reverse=True)
    return clusters
