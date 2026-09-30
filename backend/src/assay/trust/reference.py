"""Reference distributions for the Trust components (PRD 5.2, 9.2, 26 reference_sets).

Built once per model bundle from the training window and calibration predictions. Unbounded
quantities (distances, spreads) are converted to percentile rank against these references, so 1
means "among the most familiar / stable" (PRD 5.2).

Novelty must mean "outside what has been seen", not "rare". Two design choices follow from the
held-out-fraud-type stress test (PRD 6.5), which showed known fraud patterns looking unfamiliar:
  * The novelty sample has FRAUD OVER-REPRESENTED, so known fraud has neighbours.
  * The distance detector is LOCAL: a point's k-th-neighbour distance is compared with its
    neighbours' own k-th-neighbour distances (the idea behind Local Outlier Factor). A sparse but
    consistent cluster, such as account takeover, then looks familiar, while an isolated point does
    not. A plain distance percentile treats every sparse region as novel.
DEVIATION FROM PRD 9.2, which names an isolation forest as an MVP detector: an isolation forest
measures rarity, so it flags known rare fraud as novel and is not used for Familiarity here. (The
ensemble still uses it as a detection feature.)

Two samples of the training window are kept:
  ref_X / nn     a RANDOM sample, faithful to the training distribution. Used for explanation
                 perturbations and SHAP backgrounds, and as the drift reference.
  novelty sample the fraud-balanced sample described above.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors

from assay.features.registry import CATEGORICAL_FEATURES, feature_names

KNN_K = 10
FRAUD_SHARE = 0.3  # share of the novelty sample drawn from verified fraud (a parameter)
EPS = 1e-6


def pct_rank(sorted_ref: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Fraction of the reference that is <= v, in [0, 1]."""
    return np.searchsorted(sorted_ref, np.asarray(v, float), side="right") / len(sorted_ref)


@dataclass
class TrustReference:
    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    ref_X: np.ndarray  # random, distribution-faithful sample (explanations, drift)
    median: np.ndarray
    nn: NearestNeighbors  # fitted on standardised ref_X
    nov_nn: NearestNeighbors  # fitted on the fraud-balanced novelty sample
    nov_kd: np.ndarray  # leave-one-out k-th-neighbour distance of each novelty-sample point
    ratio_ecdf: np.ndarray  # sorted local-distance ratios of the novelty sample
    categorical_seen: dict[str, set[float]]
    spread_ecdf: np.ndarray
    distance_ecdf: np.ndarray
    window: str = "training window"
    meta: dict = field(default_factory=dict)

    def standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def neighbours(self, X: np.ndarray, k: int) -> np.ndarray:
        """Indices into ref_X of each row's k nearest reference points."""
        return self.nn.kneighbors(self.standardise(X), n_neighbors=k)[1]

    def local_ratio(self, X: np.ndarray, k: int = KNN_K) -> np.ndarray:
        """Own k-th-neighbour distance over the neighbours' typical k-th-neighbour distance."""
        d, ind = self.nov_nn.kneighbors(self.standardise(X), n_neighbors=k)
        return (d[:, -1] + EPS) / (self.nov_kd[ind].mean(axis=1) + EPS)

    def novelty(self, X: np.ndarray) -> np.ndarray:
        """Novelty Score N in [0, 1]: the highest of the per-case detectors (PRD 9.2): the local
        distance ratio's percentile, and an unseen-category rule (a categorical value absent from
        the reference window is out-of-distribution)."""
        n = pct_rank(self.ratio_ecdf, self.local_ratio(X))
        for name in CATEGORICAL_FEATURES:
            j = self.names.index(name)
            unseen = ~np.isin(X[:, j], list(self.categorical_seen[name]))
            n = np.where(unseen, 1.0, n)
        return np.clip(n, 0.0, 1.0)


def _novelty_sample(X: np.ndarray, y: np.ndarray | None, max_ref: int, rng) -> np.ndarray:
    if y is None or not np.any(y == 1):
        return X[np.sort(rng.choice(len(X), min(max_ref, len(X)), replace=False))]
    fraud, legit = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_f = min(len(fraud), int(max_ref * FRAUD_SHARE))
    n_l = min(len(legit), max_ref - n_f)
    idx = np.concatenate([rng.choice(fraud, n_f, replace=False), rng.choice(legit, n_l, replace=False)])
    return X[np.sort(idx)]


def build_reference(X_train: np.ndarray, cal_spread: np.ndarray, cal_distance: np.ndarray,
                    *, y_train: np.ndarray | None = None, max_ref: int = 4000,
                    seed: int = 0) -> TrustReference:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), min(max_ref, len(X_train)), replace=False)
    ref = X_train[np.sort(idx)]
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    nn = NearestNeighbors(n_neighbors=KNN_K + 1).fit((ref - mean) / std)

    nov = _novelty_sample(X_train, y_train, max_ref, rng)
    zn = (nov - mean) / std
    nov_nn = NearestNeighbors(n_neighbors=KNN_K + 1).fit(zn)
    dist, ind = nov_nn.kneighbors(zn, n_neighbors=KNN_K + 1)  # column 0 is the point itself
    kd = dist[:, KNN_K]  # leave-one-out k-th neighbour distance
    ratio = (kd + EPS) / (kd[ind[:, 1:]].mean(axis=1) + EPS)
    names = feature_names()
    seen = {n: set(np.unique(X_train[:, names.index(n)]).tolist()) for n in CATEGORICAL_FEATURES}
    return TrustReference(
        names, mean, std, ref, np.median(X_train, axis=0), nn, nov_nn, kd, np.sort(ratio), seen,
        np.sort(cal_spread), np.sort(cal_distance),
        meta={"n_train": len(X_train), "n_ref": len(ref), "n_novelty_ref": len(nov), "k": KNN_K,
              "fraud_share_requested": FRAUD_SHARE if y_train is not None else 0.0})
