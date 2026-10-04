"""Cross-method consistency (PRD 7.1, V1): do two independent ways of asking "what drove this score?" agree?

Method one is the exact tree SHAP the explanations use, which explains the boosted-tree members only.
Method two is a local permutation importance on the WHOLE ensemble: replace one feature group at a time
with values drawn from the reference data, and see how far the model's score moves.

Agreement is the average of two things, both in [0, 1]: whether the two methods name the same main
drivers (Jaccard), and whether their importances point the same way (cosine of the non-negative
vectors). Rank correlation is avoided for the reason given in explain.py.

What agreement does and does not mean (PRD 7.4). The two methods look at the same model and often share
its biases, so agreement does not prove either is right. Disagreement is informative, and expected in
one known case: the permutation view includes the random forest and isolation forest, which SHAP here
does not, so a case driven by those parts will disagree by design. It is shown to investigators; it is
not part of the Trust Index, which has not been shown to benefit from it.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from assay.features.registry import group_index

from .explain import attributions, driver_mask


@dataclass(frozen=True)
class CrossMethodConfig:
    n_background: int = 12
    driver_share: float = 0.8
    k_max: int = 3
    seed: int = 0


def local_permutation_importance(model, X: np.ndarray, ref, cfg: CrossMethodConfig | None = None,
                                 txn_ids: list[str] | None = None) -> np.ndarray:
    """(n, groups): mean absolute change in the ensemble's raw score when one group is replaced with
    reference values. Deterministic for a given case id."""
    cfg = cfg or CrossMethodConfig()
    groups = list(group_index().values())
    n = len(X)
    base = model.predict(X).raw
    out = np.zeros((n, len(groups)))
    for i in range(n):
        rng = np.random.default_rng([cfg.seed, zlib.crc32((txn_ids[i] if txn_ids else str(i)).encode())])
        bg = ref.ref_X[rng.integers(0, len(ref.ref_X), cfg.n_background)]
        blocks = []
        for idx in groups:
            block = np.repeat(X[i][None, :], cfg.n_background, axis=0)
            block[:, idx] = bg[:, idx]
            blocks.append(block)
        raw = model.predict(np.concatenate(blocks)).raw.reshape(len(groups), cfg.n_background)
        out[i] = np.abs(raw - base[i]).mean(axis=1)
    return out


def agreement(shap_groups: np.ndarray, perm_groups: np.ndarray, cfg: CrossMethodConfig | None = None) -> np.ndarray:
    """Agreement in [0, 1] between a SHAP group-attribution matrix and a permutation-importance matrix."""
    cfg = cfg or CrossMethodConfig()
    a, b = np.abs(shap_groups), np.abs(perm_groups)
    da, db = driver_mask(a, cfg.driver_share, cfg.k_max), driver_mask(b, cfg.driver_share, cfg.k_max)
    jac = (da & db).sum(-1) / np.maximum((da | db).sum(-1), 1)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    zero_a, zero_b = na < 1e-12, nb < 1e-12
    cos = np.divide((a * b).sum(-1), na * nb, out=np.zeros(len(a)), where=(na * nb) > 1e-12)
    score = 0.5 * jac + 0.5 * np.clip(cos, 0.0, 1.0)
    # `driver_mask` always names at least one group, so an all-zero vector would "agree" with any vector whose
    # top driver happens to be the first group. Two methods that both see nothing agree; one that sees nothing
    # and one that sees something do not.
    return np.where(zero_a & zero_b, 1.0, np.where(zero_a ^ zero_b, 0.0, score))


def cross_method(model, X: np.ndarray, ref, txn_ids: list[str] | None = None,
                 cfg: CrossMethodConfig | None = None) -> dict:
    """Both views and their agreement for each row of X."""
    cfg = cfg or CrossMethodConfig()
    gmap = group_index()
    names = list(gmap)
    A = attributions(model.boosters, X)
    shap_g = np.stack([A[:, idx].sum(axis=1) for idx in gmap.values()], axis=1)
    perm_g = local_permutation_importance(model, X, ref, cfg, txn_ids)
    score = agreement(shap_g, perm_g, cfg)

    def drivers(M):
        mask = driver_mask(np.abs(M), cfg.driver_share, cfg.k_max)
        return [[names[j] for j in np.argsort(-np.abs(M[i])) if mask[i, j]] for i in range(len(M))]

    return {"groups": names, "agreement": score, "shap_drivers": drivers(shap_g),
            "permutation_drivers": drivers(perm_g), "shap": shap_g, "permutation": perm_g}
