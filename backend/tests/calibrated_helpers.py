"""Synthetic cases with a KNOWN probability of a correct recommendation, shared by the calibrated tests."""

import numpy as np

from assay.trust import Component, TrustConfig, compute_trust_index
from assay.trust.assessor import CaseAssessment
from assay.trust.calibrated import scoreable
from assay.trust.reliability import Cohort

TRUE_BETA = {"rel": 1.3, "conf": 0.9, "fam": 0.2, "exp": 0.0}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def make_cases(n, seed, with_exp=True, exp_weight=0.0, base=3.0, exp_share=1.0):
    """Cases with known P(correct) = sigmoid(base + b_rel*logit(rel) + b_conf*logit(conf) + ...)."""
    rng = np.random.default_rng(seed)
    comps = {"conf": rng.uniform(0.05, 0.95, n), "rel": rng.uniform(0.05, 0.95, n), "exp": rng.uniform(0.05, 0.95, n),
             "fam": rng.uniform(0.2, 0.95, n),       # not so unfamiliar that the novelty gate fires
             "drift": rng.uniform(0.6, 1.0, n), "dq": np.full(n, 0.95)}   # data quality above its floor
    z = (base + TRUE_BETA["rel"] * logit(comps["rel"]) + TRUE_BETA["conf"] * logit(comps["conf"])
         + TRUE_BETA["fam"] * logit(comps["fam"]) + exp_weight * logit(comps["exp"]))
    p_true = sigmoid(z)
    correct = (rng.uniform(size=n) < p_true).astype(int)
    bands = rng.choice(["low", "medium", "high"], n)
    cases = []
    for i in range(n):
        c = {k: Component.active(float(comps[k][i]), n=1) for k in ("conf", "fam", "drift", "dq")}
        c["rel"] = Component.active(float(comps["rel"][i]), n=120)
        c["hum"] = Component.inactive()
        has = with_exp and rng.uniform() < exp_share
        c["exp"] = Component.active(float(comps["exp"][i]), n=3) if has else Component.missing()
        result = compute_trust_index(c, TrustConfig(max_interval_width=100.0))
        cases.append(CaseAssessment(f"t{i}", result, c, 0.1, 0.0, {"q": 1.0},
                                    Cohort("default", "card", "mid", str(bands[i]), "b-1"), {}))
    assert all(scoreable(a) for a in cases), "the generator must produce scoreable cases"
    return cases, correct, p_true
