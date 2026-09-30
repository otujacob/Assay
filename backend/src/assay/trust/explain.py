"""Explanation Reliability (PRD 7, FR-11 to FR-13).

Attributions are exact tree SHAP values (XGBoost's built-in TreeSHAP) averaged over the bootstrap
boosters, in log-odds. LIMITATION: they explain the boosted-tree members only (the largest part
of the blend), not the random forest or isolation forest.

Everything is scored at feature-GROUP level (PRD 7.4: attributions for correlated features are
unstable by construction). The "main drivers" of a case are the smallest set of groups covering
`driver_share` of the total absolute attribution (at most `k_max`).

Sub-scores, each in [0, 1]:
  stability       main-driver sets (Jaccard) and attribution vectors (cosine) under small,
                  plausible input changes. Deliberate deviation from the PRD wording "rank
                  correlation": rank correlation over-weights the ordering of negligible
                  attributions, which made stable and unstable models indistinguishable in testing.
  sensitivity     does the explanation change more than the input change warrants? Only
                  perturbations that leave the prediction unchanged count (PRD 7.1).
  faithfulness    deletion test: replacing the named drivers with background values should move the
                  model in the stated direction and by more than replacing random groups. The
                  baseline is the background distribution (what SHAP measures against), not the
                  median, which is not neutral if it sits on the same side of the decision boundary.
  reproducibility regenerating from the stored model gives the same attributions.
exp is the geometric mean of the available sub-scores (PRD 7.2), so one poor result is not hidden.

Stable is not the same as correct (PRD 7.4): a model can give a stable, faithful explanation of a
wrong prediction. Whether low exp predicts errors is hypothesis H3, tested by the harness.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from assay.features.registry import group_index

from .reference import TrustReference


@dataclass(frozen=True)
class ExplainConfig:
    m: int = 20  # perturbations per case (OPD-7 working default)
    driver_share: float = 0.8  # drivers cover this share of total |attribution|
    k_max: int = 3
    neighbours: int = 10
    swap_prob: float = 0.35  # chance each feature group is swapped with a neighbour's value
    kappa: float = 2.0  # tolerated ratio of explanation change to input change
    tau_margin: float = 0.5  # "prediction unchanged" if |delta log-odds| < tau
    n_random: int = 6  # random group sets compared in the deletion test
    n_background: int = 8  # background rows averaged in the deletion test
    seed: int = 0
    chunk: int = 300
    repro_tol: float = 1e-9


@dataclass
class ExplanationResult:
    attributions: np.ndarray  # (n, d) per-feature TreeSHAP, log-odds
    group_names: tuple[str, ...]
    group_attr: np.ndarray  # (n, g)
    stability: np.ndarray  # nan where unavailable
    sensitivity: np.ndarray
    faithfulness: np.ndarray
    reproducible: np.ndarray  # bool
    exp: np.ndarray
    exp_lo: np.ndarray
    exp_hi: np.ndarray
    n_valid_perturbations: np.ndarray


def _margin(boosters, X: np.ndarray) -> np.ndarray:
    d = xgb.DMatrix(X)
    return np.mean([b.get_booster().predict(d, output_margin=True) for b in boosters], axis=0)


def _contribs(boosters, X: np.ndarray) -> np.ndarray:
    d = xgb.DMatrix(X)
    return np.mean([b.get_booster().predict(d, pred_contribs=True) for b in boosters], axis=0)[:, :-1]


def attributions(boosters, X: np.ndarray) -> np.ndarray:
    return _contribs(boosters, X)


def _group_sum(A: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    return np.stack([A[..., idx].sum(axis=-1) for idx in groups], axis=-1)


def driver_mask(G: np.ndarray, share: float, k_max: int) -> np.ndarray:
    """Boolean mask of the main-driver groups: smallest set covering `share` of total |G|."""
    a = np.abs(G)
    order = np.argsort(-a, axis=-1, kind="stable")
    sorted_a = np.take_along_axis(a, order, axis=-1)
    total = sorted_a.sum(-1, keepdims=True)
    before = np.cumsum(sorted_a, axis=-1) - sorted_a
    include = (before < share * total) & (np.arange(a.shape[-1]) < k_max)
    include[..., 0] = True  # always at least the largest driver
    mask = np.zeros(G.shape, dtype=bool)
    np.put_along_axis(mask, order, include, axis=-1)
    return mask


def _cosine01(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity mapped to [0, 1]; two all-zero vectors count as identical."""
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    den = na * nb
    cos = np.divide((a * b).sum(-1), den, out=np.ones(den.shape), where=den > 1e-12)
    return (np.clip(cos, -1.0, 1.0) + 1.0) / 2.0


def _case_rng(seed: int, txn_id: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(txn_id.encode())])


def _geo(parts: list[np.ndarray]) -> np.ndarray:
    """Geometric mean over the sub-scores that are available (not NaN) for each case."""
    P = np.stack(parts, axis=-1)
    avail = ~np.isnan(P)
    logs = np.where(avail, np.log(np.maximum(np.nan_to_num(P, nan=1.0), 0.01)), 0.0)
    n = avail.sum(-1)
    return np.where(n > 0, np.exp(logs.sum(-1) / np.maximum(n, 1)), np.nan)


def explain_batch(boosters, X: np.ndarray, txn_ids: list[str], ref: TrustReference,
                  cfg: ExplainConfig | None = None) -> ExplanationResult:
    cfg = cfg or ExplainConfig()
    gmap = group_index()
    gnames = tuple(gmap)
    groups = [gmap[g] for g in gnames]
    ng = len(groups)
    n, d = X.shape
    A = _contribs(boosters, X)
    repro = np.abs(A - _contribs(boosters, X)).max(axis=1) <= cfg.repro_tol
    G = _group_sum(A, groups)
    base_margin = _margin(boosters, X)

    stab_pm = np.full((n, cfg.m), np.nan)
    sens_pm = np.full((n, cfg.m), np.nan)
    faith = np.full(n, np.nan)
    neigh_all = ref.neighbours(X, cfg.neighbours)
    gmask_feat = np.zeros((ng, d), dtype=bool)
    for gi, idx in enumerate(groups):
        gmask_feat[gi, idx] = True

    for lo in range(0, n, cfg.chunk):
        hi = min(n, lo + cfg.chunk)
        Xc, Gc, mc = X[lo:hi], G[lo:hi], base_margin[lo:hi]
        c = hi - lo

        # --- stability & sensitivity: swap whole groups with a random near neighbour's values ---
        Xp = np.repeat(Xc[:, None, :], cfg.m, axis=1)
        for i in range(c):
            rng = _case_rng(cfg.seed, txn_ids[lo + i])
            pick = neigh_all[lo + i][rng.integers(0, cfg.neighbours, cfg.m)]
            swap = rng.random((cfg.m, ng)) < cfg.swap_prob
            for gi, idx in enumerate(groups):
                rows = np.flatnonzero(swap[:, gi])
                if len(rows):
                    Xp[i][np.ix_(rows, idx)] = ref.ref_X[np.ix_(pick[rows], idx)]
        flat = Xp.reshape(c * cfg.m, d)
        Gp = _group_sum(_contribs(boosters, flat).reshape(c, cfg.m, d), groups)
        mp = _margin(boosters, flat).reshape(c, cfg.m)
        valid = np.abs(mp - mc[:, None]) < cfg.tau_margin  # prediction unchanged

        D0 = driver_mask(Gc, cfg.driver_share, cfg.k_max)[:, None, :]
        Dp = driver_mask(Gp, cfg.driver_share, cfg.k_max)
        jaccard = (D0 & Dp).sum(-1) / np.maximum((D0 | Dp).sum(-1), 1)
        stab = 0.5 * jaccard + 0.5 * _cosine01(np.broadcast_to(Gc[:, None, :], Gp.shape), Gp)
        r = np.abs(Gp - Gc[:, None, :]).sum(-1) / (np.abs(Gc).sum(-1)[:, None] + 1e-9)
        cin = (np.abs(Xp - Xc[:, None, :]) / ref.std).mean(-1)
        sens = 1.0 - np.clip(r - cfg.kappa * cin, 0.0, 1.0)
        stab_pm[lo:hi] = np.where(valid, stab, np.nan)
        sens_pm[lo:hi] = np.where(valid, sens, np.nan)

        # --- faithfulness: replace the named drivers vs random groups with background values ---
        drivers = driver_mask(Gc, cfg.driver_share, cfg.k_max)
        expected = -(Gc * drivers).sum(-1)  # stated effect of removing the drivers
        blocks = []
        for i in range(c):
            rng = _case_rng(cfg.seed + 1, txn_ids[lo + i])
            bg = ref.ref_X[rng.integers(0, len(ref.ref_X), cfg.n_background)]
            k = int(drivers[i].sum())
            sels = [drivers[i]]
            for _ in range(cfg.n_random):
                s = np.zeros(ng, dtype=bool)
                s[rng.choice(ng, k, replace=False)] = True
                sels.append(s)
            block = np.empty((len(sels), cfg.n_background, d))
            for si, s in enumerate(sels):
                block[si] = np.where(gmask_feat[s].any(axis=0), bg, Xc[i])
            blocks.append(block.reshape(-1, d))
        moved = _margin(boosters, np.concatenate(blocks))
        moved = moved.reshape(c, 1 + cfg.n_random, cfg.n_background).mean(axis=2) - mc[:, None]
        move_top = np.maximum(moved[:, 0] * np.sign(expected), 0.0)
        move_rand = np.abs(moved[:, 1:]).mean(axis=1)
        denom = move_top + move_rand
        ok = (denom > 1e-6) & (np.abs(expected) > 1e-9)
        faith[lo:hi] = np.where(ok, move_top / np.maximum(denom, 1e-12), np.nan)

    with np.errstate(all="ignore"):
        stability = np.nanmean(stab_pm, axis=1)
        sensitivity = np.nanmean(sens_pm, axis=1)
        s_p10, s_p90 = np.nanpercentile(stab_pm, [10, 90], axis=1)
        v_p10, v_p90 = np.nanpercentile(sens_pm, [10, 90], axis=1)
    reproducibility = np.where(repro, 1.0, 0.0)
    exp = _geo([stability, sensitivity, faith, reproducibility])
    exp_lo = _geo([s_p10, v_p10, faith, reproducibility])
    exp_hi = _geo([s_p90, v_p90, faith, reproducibility])
    return ExplanationResult(A, gnames, G, stability, sensitivity, faith, repro, exp,
                             np.minimum(exp_lo, exp), np.maximum(exp_hi, exp),
                             (~np.isnan(stab_pm)).sum(axis=1))
