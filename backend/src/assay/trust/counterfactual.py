"""Counterfactuals: what small, realistic change would have flipped the model's call? (PRD 7.1, V1)

For a case the model flags, this looks for the fewest changes to the transaction's own properties that
would make the model NOT flag it (and the reverse for a case that is not flagged). Every candidate is
re-scored by the real model, so a counterfactual is a fact about the model, not an estimate.

IMPORTANT (PRD 7.4). A counterfactual describes the MODEL, not cause and effect. "The model would not
have flagged this at 02:00" does not mean the customer should have transacted at 02:00, or that fraud
depends on the hour. It must never be presented to customers as advice. It is for investigators, to
see what the model is leaning on.

How it works:
  * Changes are made to whole feature GROUPS (the registry's groups) so no impossible combination is
    ever produced, for example an amount whose log disagrees with it.
  * Some groups cannot be changed in a "what if": the customer's transaction history (velocity) and
    the data's completeness. They stay fixed (`immutable_groups`).
  * Candidate values come from the reference data's own range, so they are realistic.
  * One group is tried first, then pairs of groups (and optionally a third), and the flips are ranked
    by how few groups changed and how far the values moved. Pairs are shortlisted by the boosted trees
    alone, which is fast, and only the shortlist is re-scored by the full model. The full model is slow
    (it is single-threaded so replays are exact), and every counterfactual that is reported was scored
    by it. The shortlist can miss a flip the boosted trees rank low; it never reports one that is not.
  * A counterfactual is MINIMAL: if undoing any one of its changes still flips the call, it is dropped,
    because that change was not doing anything.

A counterfactual is only VALID if an independent check (`validate_counterfactual`, which does not
trust the generator) finds that it flips the decision by a margin, stays inside the data's bounds,
leaves immutable features alone, is internally consistent, changes few groups, moves a small distance,
and keeps flipping under small perturbations (the flip is not an accident of a knife edge).
"""

from __future__ import annotations

import itertools
import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from assay.features.registry import (
    CATEGORICAL_FEATURES,
    CHANNEL_ORDER,
    FEATURE_GROUPS,
    feature_names,
)

from .reference import TrustReference

NAMES = feature_names()
IDX = {n: i for i, n in enumerate(NAMES)}
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
BINARY = ("new_beneficiary", "new_device", "country_mismatch")
# Which features count towards a group's distance. `amount` stands for its redundant triple.
DISTANCE_FEATURES = {"amount": ("log_amount",), "velocity": (), "timing": ("hour",), "channel": ("channel_code",),
                     "counterparty": ("new_beneficiary", "beneficiary_shared_customers"),
                     "device_location": ("new_device", "country_mismatch"), "completeness": ()}


@dataclass(frozen=True)
class CounterfactualConfig:
    max_changes: int = 2              # at most this many groups changed (parameter)
    n_return: int = 3
    flip_margin: float = 0.10         # the new risk must clear the threshold by this share of it (parameter)
    immutable_groups: tuple[str, ...] = ("velocity", "completeness")
    amount_factors: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 1.5, 2.0, 4.0)
    amount_quantiles: tuple[float, ...] = tuple(np.round(np.linspace(0.03, 0.97, 21), 3))
    max_combo_candidates: int = 12
    max_distance: float = 4.0         # "small": a normalised distance over the changed features (parameter)
    n_robust: int = 20
    noise: float = 0.05               # relative noise applied to amount (parameter)
    robust_min: float = 0.8           # share of perturbations that must keep the flip (parameter)
    beam: int = 40                    # candidates extended to a third group when max_changes is 3
    shortlist: int = 200              # pairs re-scored by the full model, picked by the boosted trees (parameter)
    seed: int = 0


@dataclass
class Counterfactual:
    x: np.ndarray
    groups: tuple[str, ...]
    changes: list[dict]
    risk_before: float
    risk_after: float
    distance: float
    checks: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return bool(self.checks) and all(v is True for k, v in self.checks.items() if k != "robust_share")


# ---------------------------------------------------------------------------------------------------
def _flag(risk: np.ndarray, t_high: float) -> np.ndarray:
    return np.asarray(risk) >= t_high


def _flips(risk: np.ndarray, was_flagged: bool, t_high: float, margin: float) -> np.ndarray:
    """Cleared the threshold by a margin, in the direction that flips the call."""
    return risk <= t_high * (1 - margin) if was_flagged else risk >= t_high * (1 + margin)


def _set_amount(x: np.ndarray, new_amount: float) -> np.ndarray:
    """Change the amount and its two derived features together, so the group stays coherent. The ratio to the
    customer's baseline scales with the amount, except when it is the 'fewer than 3 prior' placeholder 1.0."""
    out = x.copy()
    old = max(float(x[IDX["amount"]]), 1e-9)
    ratio = float(x[IDX["amount_ratio_baseline"]])
    out[IDX["amount"]] = new_amount
    out[IDX["log_amount"]] = math.log1p(new_amount)
    out[IDX["amount_ratio_baseline"]] = ratio if abs(ratio - 1.0) < 1e-9 else ratio * new_amount / old
    return out


def _set_features(x: np.ndarray, names: tuple[str, ...], values) -> np.ndarray:
    out = x.copy()
    for n, v in zip(names, values, strict=True):
        out[IDX[n]] = v
    return out


def _combos(ref: TrustReference, names: tuple[str, ...], x: np.ndarray, cap: int) -> list[tuple[float, ...]]:
    """The most common value combinations of a group in the reference data, other than the case's own."""
    cols = [IDX[n] for n in names]
    rows, counts = np.unique(np.round(ref.ref_X[:, cols], 6), axis=0, return_counts=True)
    own = tuple(np.round(x[cols], 6))
    ranked = [tuple(r) for _, r in sorted(zip(-counts, [tuple(r) for r in rows], strict=True))]
    return [r for r in ranked if r != own][:cap]


def candidates(x: np.ndarray, ref: TrustReference, cfg: CounterfactualConfig) -> dict[str, list[np.ndarray]]:
    """Realistic single-group variants of the case, each a full feature vector."""
    out: dict[str, list[np.ndarray]] = {}
    mutable = [g for g in FEATURE_GROUPS if g not in cfg.immutable_groups]
    if "amount" in mutable:
        amt = float(x[IDX["amount"]])
        grid = set(np.quantile(ref.ref_X[:, IDX["amount"]], cfg.amount_quantiles).round(2).tolist())
        grid |= {round(amt * f, 2) for f in cfg.amount_factors}
        out["amount"] = [_set_amount(x, a) for a in sorted(grid) if a > 0 and abs(a - amt) > 1e-6]
    if "timing" in mutable:
        out["timing"] = [_set_features(x, ("hour",), (h,)) for h in range(24) if h != int(x[IDX["hour"]])]
    if "channel" in mutable:
        seen = sorted(ref.categorical_seen["channel_code"])
        out["channel"] = [_set_features(x, ("channel_code",), (c,)) for c in seen if c != x[IDX["channel_code"]]]
    # Every other mutable group (counterparty, device_location, and velocity or completeness if the institution
    # opened them up) takes the most common value combinations the reference data actually contains.
    for g in mutable:
        if g not in out:
            names = FEATURE_GROUPS[g]
            out[g] = [_set_features(x, names, v) for v in _combos(ref, names, x, cfg.max_combo_candidates)]
    return {g: v for g, v in out.items() if v}


def _distance(x: np.ndarray, cf: np.ndarray, ref: TrustReference, groups) -> float:
    total = 0.0
    for g in groups:
        for f in DISTANCE_FEATURES[g]:
            j = IDX[f]
            if f in CATEGORICAL_FEATURES:
                total += 1.0 if x[j] != cf[j] else 0.0
            elif f == "hour":
                d = abs(x[j] - cf[j])
                total += min(d, 24 - d) / 6.0           # circular, in units of six hours
            else:
                total += abs(x[j] - cf[j]) / max(float(ref.std[j]), 1e-9)
    return float(total)


def _changed(x: np.ndarray, cf: np.ndarray) -> list[dict]:
    return [{"feature": n, "from": float(x[i]), "to": float(cf[i])} for i, n in enumerate(NAMES)
            if abs(x[i] - cf[i]) > 1e-9]


# ---------------------------------------------------------------------------------------------------
def validate_many(model, ref: TrustReference, x: np.ndarray, cfs: list[np.ndarray], t_high: float,
                  cfg: CounterfactualConfig | None = None, *, groups: list | None = None) -> list[dict]:
    """Independent checks of counterfactuals (PRD 7.1), all scored in ONE call to the model (it is slow).
    Does not trust whatever produced them: it re-scores and re-derives everything."""
    cfg = cfg or CounterfactualConfig()
    n_rows = len(cfs)
    noisy_all, eps_all = [], []
    for cf in cfs:
        rng = np.random.default_rng([cfg.seed, zlib.crc32(cf.tobytes()) & 0xFFFFFFFF])
        eps = rng.normal(0.0, cfg.noise, cfg.n_robust)
        amt = float(cf[IDX["amount"]])
        if math.isfinite(amt) and amt > 0:
            noisy_all.append(np.stack([_set_amount(cf, max(amt * (1 + e), 0.01)) for e in eps]))
        else:                       # an invalid amount is rejected below; there is nothing meaningful to wobble
            noisy_all.append(np.repeat(cf[None, :], cfg.n_robust, axis=0))
        eps_all.append(eps)
    risks = model.predict(np.concatenate([x[None, :], np.stack(cfs), *noisy_all])).calibrated
    risk0, risk1, risk_noisy = float(risks[0]), risks[1:1 + n_rows], risks[1 + n_rows:].reshape(n_rows, cfg.n_robust)
    was_flagged = risk0 >= t_high
    lo, hi = ref.ref_X.min(axis=0), ref.ref_X.max(axis=0)
    out = []
    for k, cf in enumerate(cfs):
        changed_features = [NAMES[i] for i in range(len(NAMES)) if abs(x[i] - cf[i]) > 1e-9]
        changed_groups = sorted({GROUP_OF[f] for f in changed_features}) if groups is None else sorted(groups[k])
        in_bounds = True
        for f in changed_features:
            j = IDX[f]
            if f in CATEGORICAL_FEATURES:
                in_bounds &= bool(cf[j] in ref.categorical_seen[f])
            elif f in BINARY:
                in_bounds &= bool(cf[j] in (0.0, 1.0))
            elif f not in ("amount", "log_amount", "amount_ratio_baseline"):
                in_bounds &= bool(lo[j] - 1e-9 <= cf[j] <= hi[j] + 1e-9)
        # An amount may lie outside the reference range (a much larger or smaller one is a fair what-if), but it
        # must be positive and finite, and the whole group must stay coherent (below).
        amt = float(cf[IDX["amount"]])
        in_bounds &= bool(math.isfinite(amt) and amt > 0)
        immutable_ok = all(abs(x[IDX[f]] - cf[IDX[f]]) < 1e-9 for g in cfg.immutable_groups for f in FEATURE_GROUPS[g])
        coherent = bool(
            abs(cf[IDX["log_amount"]] - math.log1p(max(amt, 0))) < 1e-6
            and cf[IDX["amount_ratio_baseline"]] >= 0
            and cf[IDX["hour"]] == int(cf[IDX["hour"]]) and 0 <= cf[IDX["hour"]] <= 23
            and cf[IDX["missing_fields"]] == int(cf[IDX["missing_fields"]]))
        dist = _distance(x, cf, ref, changed_groups)
        # robustness: the flip must survive a small relative change to the amount (not a knife edge)
        kept = (risk_noisy[k] < t_high) if was_flagged else (risk_noisy[k] >= t_high)
        robust_share = float(np.mean(kept))
        out.append({
            "flips": bool(_flips(np.array([risk1[k]]), was_flagged, t_high, cfg.flip_margin)[0]),
            "within_bounds": bool(in_bounds),
            "immutable_unchanged": bool(immutable_ok),
            "coherent": coherent,
            "few_changes": bool(0 < len(changed_groups) <= cfg.max_changes),
            "small_change": bool(dist <= cfg.max_distance),
            "robust": bool(robust_share >= cfg.robust_min),
            "robust_share": robust_share,
        })
    return out


def validate_counterfactual(model, ref: TrustReference, x: np.ndarray, cf: np.ndarray, t_high: float,
                            cfg: CounterfactualConfig | None = None, *, groups=None) -> dict:
    """One counterfactual's checks: {check: bool, ..., "robust_share": float}. See `validate_many`."""
    return validate_many(model, ref, x, [cf], t_high, cfg, groups=None if groups is None else [groups])[0]


# ---------------------------------------------------------------------------------------------------
def generate(model, ref: TrustReference, x: np.ndarray, t_high: float,
             cfg: CounterfactualConfig | None = None) -> list[Counterfactual]:
    """Counterfactuals for one case: the ones that flip the model's call with the fewest, smallest changes,
    best first. May be empty: some cases have no realistic flip, which is information, not a failure."""
    cfg = cfg or CounterfactualConfig()
    x = np.asarray(x, dtype=float)
    risk0 = float(model.predict(x[None, :]).calibrated[0])
    was_flagged = risk0 >= t_high
    cands = candidates(x, ref, cfg)
    groups = sorted(cands)

    def merge(base: np.ndarray, group: str, variant: np.ndarray) -> np.ndarray:
        out = base.copy()
        for n in FEATURE_GROUPS[group]:
            out[IDX[n]] = variant[IDX[n]]
        return out

    def proxy(X: np.ndarray) -> np.ndarray:
        """Cheap ranking score: the boosted trees' mean probability. Used only to choose what to re-score."""
        return np.mean([b.predict_proba(X)[:, 1] for b in model.boosters], axis=0)

    singles = [(c, (g,)) for g in groups for c in cands[g]]
    if not singles:
        return []
    pool = list(singles)
    risk = model.predict(np.stack([c for c, _ in singles])).calibrated
    if cfg.max_changes >= 2:
        pairs = [(merge(c1, g2, c2), (g1, g2)) for g1, g2 in itertools.combinations(groups, 2)
                 for c1 in cands[g1] for c2 in cands[g2]]
        if pairs:
            Xp = np.stack([c for c, _ in pairs])
            score = proxy(Xp)
            keep = np.argsort(score if was_flagged else -score)[: cfg.shortlist]
            short = [pairs[k] for k in keep]
            risk = np.concatenate([risk, model.predict(np.stack([c for c, _ in short])).calibrated])
            pool += short
    flip = _flips(risk, was_flagged, t_high, cfg.flip_margin)

    if cfg.max_changes >= 3 and len(groups) >= 3:
        order = np.argsort(risk if was_flagged else -risk)
        seeds = [k for k in order if len(pool[k][1]) == 2][: cfg.beam]
        extra = [(merge(pool[k][0], g3, c3), (*pool[k][1], g3)) for k in seeds for g3 in groups
                 if g3 not in pool[k][1] for c3 in cands[g3]]
        if extra:
            re_ = model.predict(np.stack([c for c, _ in extra])).calibrated
            pool += extra
            risk = np.concatenate([risk, re_])
            flip = np.concatenate([flip, _flips(re_, was_flagged, t_high, cfg.flip_margin)])

    def revert(vec: np.ndarray, group: str) -> np.ndarray:
        out = vec.copy()
        for n in FEATURE_GROUPS[group]:
            out[IDX[n]] = x[IDX[n]]
        return out

    ranked = sorted((len(pool[k][1]), _distance(x, pool[k][0], ref, pool[k][1]), k) for k in np.flatnonzero(flip))
    # Minimality: drop a counterfactual if undoing any one of its changes still flips the call.
    head = ranked[: cfg.n_return * 12]
    reversions = [(pos, revert(pool[k][0], g)) for pos, (_, _, k) in enumerate(head) for g in pool[k][1] if len(pool[k][1]) > 1]
    redundant: set[int] = set()
    if reversions:
        r_rev = model.predict(np.stack([v for _, v in reversions])).calibrated
        for (pos, _), f in zip(reversions, _flips(r_rev, was_flagged, t_high, cfg.flip_margin), strict=True):
            if f:
                redundant.add(pos)
    chosen: list[Counterfactual] = []
    seen_sets: set[tuple[str, ...]] = set()
    for pos, (_, dist, k) in enumerate(head):       # one per distinct set of changed groups
        vec, used = pool[k]
        if pos in redundant or tuple(sorted(used)) in seen_sets:
            continue
        seen_sets.add(tuple(sorted(used)))
        chosen.append(Counterfactual(vec, tuple(sorted(used)), _changed(x, vec), risk0, float(risk[k]), dist))
        if len(chosen) >= cfg.n_return * 3:         # a few more than needed, since some will fail validation
            break
    out = list(chosen)
    if out:
        for c, checks in zip(out, validate_many(model, ref, x, [c.x for c in out], t_high, cfg,
                                                groups=[c.groups for c in out]), strict=True):
            c.checks = checks
    out.sort(key=lambda c: (not c.valid, len(c.groups), c.distance))
    return out[: cfg.n_return]


# ---------------------------------------------------------------------------------------------------
def describe_change(feature: str, frm: float, to: float) -> str:
    """One change in words an investigator can read."""
    def n(v: float) -> str:
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}".rstrip("0").rstrip(".")

    if feature == "amount":
        return f"amount {n(frm)} → {n(to)}"
    if feature == "hour":
        return f"hour of day {int(frm):02d}:00 → {int(to):02d}:00"
    if feature == "channel_code":
        def nm(v):
            return CHANNEL_ORDER[int(v)] if 0 <= int(v) < len(CHANNEL_ORDER) else "other"
        return f"channel {nm(frm)} → {nm(to)}"
    if feature in BINARY:
        label = {"new_beneficiary": "a new beneficiary", "new_device": "a new device", "country_mismatch": "an unusual country"}[feature]
        return f"{label}: {'yes' if frm else 'no'} → {'yes' if to else 'no'}"
    if feature == "beneficiary_shared_customers":
        return f"other customers paying this beneficiary {n(frm)} → {n(to)}"
    return f"{feature} {n(frm)} → {n(to)}"


def summarise(cfs: list[Counterfactual]) -> dict:
    """The investigator's view. `score` is the share of returned counterfactuals that are valid and robust
    (None when none was found: that says the call is robust to realistic changes, not that anything is wrong)."""
    if not cfs:
        return {"found": 0, "score": None}
    ok = [c for c in cfs if c.valid]
    robust = float(np.mean([c.checks["robust_share"] for c in cfs]))
    return {"found": len(cfs), "valid": len(ok), "score": float(len(ok) / len(cfs) * robust)}


NOTE = ("A counterfactual describes the MODEL, not cause and effect. It shows what the model is leaning on, for an "
        "investigator. It is not advice and must not be given to a customer (PRD 7.4).")
DERIVED = ("log_amount", "amount_ratio_baseline")   # follow from the amount, so they are not listed as changes of their own


def view(cfs: list[Counterfactual], x: np.ndarray, risk: float, t_high: float) -> dict:
    """The investigator's view of a case's counterfactuals, ready to send to the screen."""
    out = []
    for c in cfs:
        # amount's two derived features follow from it, so they are not listed as changes of their own
        shown = [ch for ch in c.changes if not (ch["feature"] in DERIVED and any(k["feature"] == "amount" for k in c.changes))]
        out.append({
            "groups": list(c.groups),
            "changes": [{**ch, "text": describe_change(ch["feature"], ch["from"], ch["to"])} for ch in shown],
            "risk_before": c.risk_before, "risk_after": c.risk_after, "distance": c.distance,
            "valid": c.valid, "robust_share": c.checks.get("robust_share"),
            "failed_checks": sorted(k for k, v in c.checks.items() if v is False),
        })
    return {"risk": risk, "threshold": t_high, "flagged": risk >= t_high,
            "counterfactuals": out, "summary": summarise(cfs), "note": NOTE}
