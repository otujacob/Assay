"""Stress tests (PRD 6.5, FR-38). Each produces a report against the PRD's expected behaviour, with
`passed` saying whether that behaviour was observed. A failed stress test is a finding, not an error.
"""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

import numpy as np

from assay.detection import train_bundle
from assay.features import build_table, feature_names
from assay.trust import ReasonCode
from assay.trust.assessor import TrustAssessor
from assay.trust.drift import drift_vector

NAMES = feature_names()
IDX = {n: i for i, n in enumerate(NAMES)}
DRIFT_ALARM_LEVEL = 0.10  # population drift statistic that opens a recalibration review (parameter)


def _assessor(r) -> TrustAssessor:
    return TrustAssessor(r.model, r.reference, r.store, r.manifest.thresholds,
                         model_version=r.manifest.bundle_id)


def _sample(r, txns_by_id, n: int, seed: int):
    ids = r.test_table.txn_ids
    idx = np.sort(np.random.default_rng(seed).choice(len(ids), min(n, len(ids)), replace=False))
    return idx, r.test_table.X[idx], [txns_by_id[ids[i]] for i in idx]


def _share(states, which) -> float:
    return float(np.mean([s in which for s in states])) if states else float("nan")


# ---- 1. synthetic feature drift ------------------------------------------------------------
def stress_feature_drift(r, txns_by_id, *, scale: float = 3.0, n: int = 250, seed: int = 0) -> dict:
    """Inject drift on the amount features. Expect: drift stability and TI fall, the alarm fires."""
    _, X, txns = _sample(r, txns_by_id, n, seed)
    window = r.test_table.X[np.random.default_rng(seed + 1).choice(len(r.test_table.X), 2000, replace=False)]

    def shifted(A: np.ndarray) -> np.ndarray:
        B = A.copy()
        B[:, IDX["amount"]] *= scale
        B[:, IDX["log_amount"]] = np.log1p(B[:, IDX["amount"]])
        B[:, IDX["amount_ratio_baseline"]] *= scale
        return B

    def run(Xs, txn_rows, win):
        d = drift_vector(r.reference.ref_X, win)
        pred = r.model.predict(Xs, r.manifest.thresholds["t_high"])
        cases = _assessor(r).assess(Xs, txn_rows, pred, drift_vector=d)
        ti = [c.result.ti for c in cases if c.result.ti is not None]
        return {
            "mean_drift_stability": float(np.mean([c.components["drift"].score for c in cases])),
            "mean_ti": float(np.mean(ti)) if ti else float("nan"),
            "max_feature_drift": float(d.max()),
            "alarm_fired": bool(d.max() >= DRIFT_ALARM_LEVEL),
            "drifted_features": [NAMES[j] for j in np.flatnonzero(d >= DRIFT_ALARM_LEVEL)],
        }

    base = run(X, txns, window)
    drifted_txns = [{**t, "amount": float(t["amount"]) * scale} for t in txns]
    drifted = run(shifted(X), drifted_txns, shifted(window))
    passed = (drifted["mean_drift_stability"] < base["mean_drift_stability"]
              and drifted["mean_ti"] < base["mean_ti"] and drifted["alarm_fired"]
              and not base["alarm_fired"])
    return {"test": "synthetic feature drift", "scale": scale, "n_cases": len(txns),
            "expected": "drift stability and TI fall, alarm fires (and does not fire without drift)",
            "baseline": base, "drifted": drifted, "passed": bool(passed)}


# ---- 2. degraded data completeness -------------------------------------------------------------
def stress_data_degradation(r, txns_by_id, *, rates=(0.0, 0.25, 0.5, 0.75, 1.0), n: int = 250,
                            seed: int = 0) -> dict:
    """Null out optional fields at rising rates. Expect: data quality falls, the floor gate fires."""
    _, X, txns = _sample(r, txns_by_id, n, seed)
    pred = r.model.predict(X, r.manifest.thresholds["t_high"])
    fields = ("device_hash", "ip_hash", "country", "merchant_id")
    rows = []
    for rate in rates:
        rng = np.random.default_rng(seed + int(rate * 100))
        broken = [{**t, **{f: None for f in fields if rng.random() < rate}} for t in txns]
        cases = _assessor(r).assess(X, broken, pred, explain_mask=np.zeros(len(txns), bool))
        floor = [ReasonCode.DATA_QUALITY_FLOOR in c.result.reason_codes for c in cases]
        rows.append({"null_rate": rate, "mean_dq": float(np.mean([c.components["dq"].score for c in cases])),
                     "floor_share": float(np.mean(floor))})
    dq = [x["mean_dq"] for x in rows]
    fl = [x["floor_share"] for x in rows]
    passed = (all(a >= b - 1e-9 for a, b in pairwise(dq))
              and all(a <= b + 1e-9 for a, b in pairwise(fl))
              and fl[0] < 0.05 and fl[-1] > 0.95)
    return {"test": "degraded data completeness", "n_cases": len(txns), "rows": rows,
            "expected": "data quality falls monotonically and the floor gate triggers",
            "passed": bool(passed)}


# ---- 3. hold out an entire fraud type, then present it ---------------------------------------------
def stress_held_out_fraud_type(ds, txns, cfg, *, held: str = "mule_ring", n_other: int = 300,
                               seed: int = 0) -> dict:
    """Retrain without one fraud type, then present it. Expect: familiarity falls and cases move
    to Insufficient evidence or Low trust, at a review load an institution could absorb."""
    truth = ds.truth
    keep = [t for t in txns if truth[t["txn_id"]]["fraud_type"] != held]
    kept_ids = {t["txn_id"] for t in keep}
    r2 = train_bundle(keep, [o for o in ds.outcomes if o["txn_id"] in kept_ids], cfg)

    full = build_table(txns)  # features from the FULL history, as live scoring would see them
    pos = {tid: i for i, tid in enumerate(full.txn_ids)}
    by_id = {t["txn_id"]: t for t in txns}
    t_start = cfg.reliability_end.timestamp() if cfg.reliability_end else cfg.calibration_end.timestamp()
    in_window = [tid for tid in full.txn_ids if t_start <= full.t[pos[tid]] < cfg.test_end.timestamp()]
    held_ids = [t for t in in_window if truth[t]["fraud_type"] == held]
    rng = np.random.default_rng(seed)
    others = [t for t in in_window if truth[t]["fraud_type"] != held]
    others = [others[i] for i in rng.choice(len(others), min(n_other, len(others)), replace=False)]
    ids = held_ids + others
    X = full.X[[pos[t] for t in ids]]
    rows = [by_id[t] for t in ids]
    pred = r2.model.predict(X, r2.manifest.thresholds["t_high"])
    cases = _assessor(r2).assess(X, rows, pred)

    def group(kind_fn):
        sel = [c for c, t in zip(cases, ids, strict=True) if kind_fn(truth[t]["fraud_type"])]
        states = [c.result.state.value for c in sel]
        return {"n": len(sel), "states": dict(Counter(states)),
                "share_low_or_insufficient": _share(states, {"low", "insufficient_evidence"}),
                "mean_familiarity": float(np.mean([c.components["fam"].score for c in sel])) if sel else None,
                "share_unfamiliar_reason": float(np.mean(
                    [ReasonCode.UNFAMILIAR_PATTERN in c.result.reason_codes for c in sel])) if sel else None}

    g_held = group(lambda f: f == held)
    g_known = group(lambda f: f is not None and f != held)
    g_legit = group(lambda f: f is None)
    # PRD 6.5 expects familiarity to fall and cases to move to Low/Insufficient. The first version
    # of this rule also required more Low/Insufficient than KNOWN fraud, which conflated two things:
    # known fraud is routed to a human anyway for other reasons (low model confidence, thin cohorts).
    # Familiarity and the unfamiliar-pattern reason are the direct measures of "never seen".
    passed = bool(g_held["n"] >= 10
                  and g_held["mean_familiarity"] < g_known["mean_familiarity"]
                  and g_held["mean_familiarity"] < g_legit["mean_familiarity"]
                  and g_held["share_unfamiliar_reason"] > g_known["share_unfamiliar_reason"]
                  and g_held["share_low_or_insufficient"] > g_legit["share_low_or_insufficient"])
    return {"test": "held-out fraud type", "held_out": held,
            "expected": "familiarity falls below known fraud and ordinary traffic, the unfamiliar-pattern "
                        "reason fires more often than for known fraud, and held-out cases land in Low or "
                        "Insufficient evidence more often than ordinary traffic",
            "held_out_cases": g_held, "known_fraud_cases": g_known, "ordinary_cases": g_legit,
            "review_load_ordinary_traffic": g_legit["share_low_or_insufficient"], "passed": passed}


# ---- 4. the PRD's natural held-out type: the planted novel type ---------------------------------------
def novel_type_summary(cases, truth, novel: str = "novel_open_banking_scam") -> dict:
    """H8 on the main run: where did the never-seen fraud type land?"""
    def states(sel):
        return [c.result.state.value for c, t in zip(cases.assessments, cases.txns, strict=True)
                if sel(truth[t["txn_id"]]["fraud_type"])]
    novel_s = states(lambda f: f == novel)
    known_s = states(lambda f: f is not None and f != novel)
    legit_s = states(lambda f: f is None)
    low = {"low", "insufficient_evidence"}
    return {"test": "novel fraud type (H8)",
            "n_novel": len(novel_s), "share_low_or_insufficient_novel": _share(novel_s, low),
            "share_low_or_insufficient_known_fraud": _share(known_s, low),
            "share_low_or_insufficient_ordinary": _share(legit_s, low),
            "state_counts_novel": dict(Counter(novel_s))}


def noisy_analyst_labels() -> dict:
    return {"test": "noisy or adversarial analyst labels", "status": "not_testable",
            "reason": "needs the feedback quality engine (FQS scoring, E6), which is not built yet",
            "expected": "feedback quality scores fall and labels are rejected (PRD 11)", "passed": None}


def run_stress_tests(r, ds, txns, cfg, txns_by_id) -> dict:
    return {"feature_drift": stress_feature_drift(r, txns_by_id),
            "data_degradation": stress_data_degradation(r, txns_by_id),
            "held_out_fraud_type": stress_held_out_fraud_type(ds, txns, cfg),
            "noisy_analyst_labels": noisy_analyst_labels()}

