"""Validation gates for a candidate model (PRD 12.3).

A candidate is compared with the champion on the SAME verified-outcome-only holdout: the candidate's test
window, which only verified outcomes reach (train_bundle keeps analyst labels out of everything but the
training window, PRD 11.4). Both models are scored on those rows, so a difference is the model's, not the
data's.

Tolerances are working defaults. The PRD says they are set from measured baselines and agreed with each
institution (OPD-12), and none have been measured. A gate that cannot be judged says so (`not_evaluated`)
instead of passing: promotion needs every gate to pass or to be explicitly waived by the approver, and the
waiver is recorded.

  G1 data         manifest valid and single-tenant, leak check passed, only matured labels, class balance
                  documented, analyst labels only from the accepted pool
  G2 performance  PR-AUC, precision and recall at each model's own operating point, overall and per channel
  G3 calibration  error within tolerance, and not worse than the champion beyond tolerance
  G4 trust        high-trust error rate and error discrimination not degraded (section 6 measures)
  G5 explanation  explanation reliability not degraded
  G6 segments     a table by channel for a person to review; never passes by itself
  G7 shadow       judged later, from live shadow scoring (lifecycle.py)
  G8 integrity    artefact signed, lineage recorded
  G9 approval     a second person signs off (lifecycle.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from assay.detection.metrics import expected_calibration_error, pr_auc, precision_recall_at
from assay.trust import TrustState
from assay.validation import measures as M
from assay.validation.harness import collect_cases, recompute_without

from .status import FAIL, NOT_EVALUATED, PASS, PENDING, REVIEW

GATES_VERSION = "gates-0"


@dataclass(frozen=True)
class GateConfig:
    perf_tol: float = 0.02  # PR-AUC, precision and recall may each fall by at most this much
    segment_min_positives: int = 15  # a channel with fewer verified frauds is shown but not judged
    segment_tol: float = 0.05  # per-channel recall may fall by at most this much
    ece_max: float = 0.05
    ece_tol: float = 0.01
    min_class: int = 20  # positives and negatives needed in the training pool
    min_high_cases: int = 50  # High-trust cases needed before the high-trust error rate is judged
    high_error_tol: float = 0.01
    discrimination_tol: float = 0.02
    exp_tol: float = 0.05
    min_explained: int = 30
    trust_sample: int = 1500  # holdout cases assessed for G4 and G5
    explain_frac: float = 0.25
    seed: int = 0


@dataclass
class Gate:
    id: str
    name: str
    status: str
    detail: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateReport:
    gates: list[Gate]
    version: str = GATES_VERSION

    def get(self, gid: str) -> Gate:
        return next(g for g in self.gates if g.id == gid)

    @property
    def failed(self) -> list[Gate]:
        return [g for g in self.gates if g.status == FAIL]

    @property
    def unresolved(self) -> list[Gate]:
        """Gates that did not pass and are not simply later steps: the approver must see these."""
        return [g for g in self.gates if g.status in (NOT_EVALUATED, REVIEW)]

    def as_dict(self) -> dict:
        return {"version": self.version, "gates": [vars(g) for g in self.gates],
                "failed": [g.id for g in self.failed], "unresolved": [g.id for g in self.unresolved]}


def champion_on(candidate, champion_bundle):
    """The champion scored on the candidate's holdout, as a TrainingResult-shaped object."""
    from assay.detection.train import TrainingResult

    X = candidate.test_table.X
    th = champion_bundle.manifest.thresholds["t_high"]
    return TrainingResult(champion_bundle.model, champion_bundle.manifest, candidate.test_table, candidate.test_y,
                          champion_bundle.model.predict(X, th), champion_bundle.model.baseline_score(X),
                          champion_bundle.reference, champion_bundle.store, None)


def _rates(r) -> dict:
    th = r.manifest.thresholds["t_high"]
    pr = precision_recall_at(r.test_y, r.test_pred.calibrated, th)
    return {"pr_auc": pr_auc(r.test_y, r.test_pred.calibrated), "precision": pr["precision"], "recall": pr["recall"],
            "ece": expected_calibration_error(r.test_y, r.test_pred.calibrated)}


def _worse(cand: float, champ: float, tol: float) -> bool:
    return bool(np.isfinite(cand) and np.isfinite(champ) and cand < champ - tol)


def g1_data(candidate, tenant_id: str, cfg: GateConfig, pool_labels: set[str] | None,
            used_labels: set[str] | None) -> Gate:
    m, problems = candidate.manifest, []
    if m.tenant_id != tenant_id or m.dataset_tenant_ids != [tenant_id]:
        problems.append("dataset is not single-tenant")
    if not m.dataset_id or not m.feature_set_version:
        problems.append("dataset manifest incomplete")
    cb = (m.extra or {}).get("class_balance") or {}
    if not cb:
        problems.append("class balance not documented")
    elif cb.get("train_pos", 0) < cfg.min_class or cb["train_n"] - cb["train_pos"] < cfg.min_class:
        problems.append("too few positives or negatives in the training pool")
    if used_labels is not None and pool_labels is not None and not used_labels <= pool_labels:
        problems.append(f"{len(used_labels - pool_labels)} analyst labels used that the pool did not accept")
    vals = {"class_balance": cb, "extra_labels_used": (m.extra or {}).get("extra_labels_used", 0)}
    if problems:
        return Gate("G1", "Data", FAIL, "; ".join(problems), vals)
    return Gate("G1", "Data", PASS, "single-tenant, leak check passed at training, matured labels only, "
                "balance documented", vals)


def g2_performance(cand, champ, cfg: GateConfig, segment_of: dict[str, str] | None) -> Gate:
    c, h = _rates(cand), _rates(champ)
    bad = [k for k in ("pr_auc", "precision", "recall") if _worse(c[k], h[k], cfg.perf_tol)]
    seg_rows, seg_bad = [], []
    if segment_of:
        ids = np.array(cand.test_table.txn_ids)
        segs = np.array([segment_of.get(t, "unknown") for t in ids])
        tc, th = cand.manifest.thresholds["t_high"], champ.manifest.thresholds["t_high"]
        for s in sorted(set(segs)):
            m = segs == s
            pos = int(cand.test_y[m].sum())
            row = {"segment": s, "n": int(m.sum()), "positives": pos}
            if pos >= cfg.segment_min_positives:
                rc = precision_recall_at(cand.test_y[m], cand.test_pred.calibrated[m], tc)["recall"]
                rh = precision_recall_at(champ.test_y[m], champ.test_pred.calibrated[m], th)["recall"]
                row |= {"recall_candidate": rc, "recall_champion": rh, "judged": True}
                if _worse(rc, rh, cfg.segment_tol):
                    seg_bad.append(s)
            else:
                row["judged"] = False  # too few verified frauds to say anything
            seg_rows.append(row)
    vals = {"candidate": c, "champion": h, "tolerance": cfg.perf_tol, "segments": seg_rows}
    if bad or seg_bad:
        why = ([f"{k} worse than champion by more than {cfg.perf_tol}" for k in bad]
               + [f"recall worse in segment {s}" for s in seg_bad])
        return Gate("G2", "Performance", FAIL, "; ".join(why), vals)
    return Gate("G2", "Performance", PASS, f"PR-AUC {c['pr_auc']:.3f} against {h['pr_auc']:.3f}; precision and "
                "recall not worse than the champion beyond tolerance", vals)


def g3_calibration(cand, champ, cfg: GateConfig) -> Gate:
    c, h = _rates(cand)["ece"], _rates(champ)["ece"]
    vals = {"ece_candidate": c, "ece_champion": h, "ece_max": cfg.ece_max, "tolerance": cfg.ece_tol}
    if c > cfg.ece_max:
        return Gate("G3", "Calibration", FAIL, f"calibration error {c:.3f} above the limit {cfg.ece_max}", vals)
    if c > h + cfg.ece_tol:
        return Gate("G3", "Calibration", FAIL, f"calibration error {c:.3f} worse than the champion's {h:.3f}", vals)
    return Gate("G3", "Calibration", PASS, f"calibration error {c:.3f} against {h:.3f}", vals)


def _trust_numbers(cases) -> dict:
    results = recompute_without(cases, None)
    ti = np.array([np.nan if r.ti is None else float(r.ti) for r in results])
    high = np.array([r.state == TrustState.HIGH for r in results])
    wrong = cases.wrong
    exp_scores = [a.components["exp"].score for a in cases.assessments
                  if "exp" in a.components and a.components["exp"].score is not None]
    return {"high_cases": int(high.sum()), "high_error": M.rate(wrong, high),
            "discrimination": M.auroc(wrong, 100.0 - np.where(np.isnan(ti), 0.0, ti)),
            "n_explained": len(exp_scores),
            "exp_median": float(np.median(exp_scores)) if exp_scores else None}


def g4_g5_trust(cand, champ, txns_by_id: dict[str, dict], cfg: GateConfig) -> tuple[Gate, Gate]:
    cc = collect_cases(cand, txns_by_id, cfg.trust_sample, cfg.seed, cfg.explain_frac)
    hc = collect_cases(champ, txns_by_id, cfg.trust_sample, cfg.seed, cfg.explain_frac)
    c, h = _trust_numbers(cc), _trust_numbers(hc)
    vals = {"candidate": c, "champion": h}
    # G4
    if c["high_cases"] < cfg.min_high_cases or h["high_cases"] < cfg.min_high_cases:
        g4 = Gate("G4", "Trust", NOT_EVALUATED,
                  f"fewer than {cfg.min_high_cases} High-trust cases to judge the high-trust error rate", vals)
    else:
        bad = []
        ce, he = c["high_error"]["value"], h["high_error"]["value"]
        if ce is not None and he is not None and ce > he + cfg.high_error_tol:
            bad.append(f"high-trust error rate {ce:.3f} above the champion's {he:.3f}")
        if _worse(c["discrimination"], h["discrimination"], cfg.discrimination_tol):
            bad.append("error discrimination worse than the champion's")
        g4 = (Gate("G4", "Trust", FAIL, "; ".join(bad), vals) if bad else
              Gate("G4", "Trust", PASS, "high-trust error rate and error discrimination not degraded", vals))
    # G5
    if c["n_explained"] < cfg.min_explained or h["n_explained"] < cfg.min_explained or None in (c["exp_median"], h["exp_median"]):
        g5 = Gate("G5", "Explanation", NOT_EVALUATED, "too few explained cases to compare reliability", vals)
    elif _worse(c["exp_median"], h["exp_median"], cfg.exp_tol):
        g5 = Gate("G5", "Explanation", FAIL, f"median explanation reliability {c['exp_median']:.2f} "
                  f"against the champion's {h['exp_median']:.2f}", vals)
    else:
        g5 = Gate("G5", "Explanation", PASS, "explanation reliability not degraded", vals)
    return g4, g5


def g6_segments(g2: Gate) -> Gate:
    rows = g2.values.get("segments") or []
    return Gate("G6", "Segment review", REVIEW,
                "a person must review results by channel; Assay makes no fairness claim (PRD 12.3)",
                {"segments": rows})


def g8_integrity(candidate, *, artefact_signed: bool, lineage_recorded: bool) -> Gate:
    problems = [m for ok, m in ((artefact_signed, "artefact not signed"),
                                (lineage_recorded, "lineage not recorded")) if not ok]
    vals = {"bundle_id": candidate.manifest.bundle_id, "dataset_id": candidate.manifest.dataset_id}
    return (Gate("G8", "Integrity", FAIL, "; ".join(problems), vals) if problems else
            Gate("G8", "Integrity", PASS, "artefact signed and lineage recorded", vals))


def evaluate(candidate, champion, *, tenant_id: str, txns_by_id: dict[str, dict],
             cfg: GateConfig | None = None, pool_labels: set[str] | None = None,
             used_labels: set[str] | None = None, segment_of: dict[str, str] | None = None,
             artefact_signed: bool = False, lineage_recorded: bool = False) -> GateReport:
    """Run G1 to G6 and G8. G7 and G9 are lifecycle steps and appear as pending."""
    cfg = cfg or GateConfig()
    if candidate.test_table.txn_ids != champion.test_table.txn_ids:
        raise ValueError("candidate and champion must be scored on the same holdout")
    if candidate.manifest.feature_names != champion.manifest.feature_names:
        g1 = Gate("G1", "Data", FAIL, "feature set differs from the champion's; compare after re-scoring "
                  "the same features", {})
        return GateReport([g1])
    g1 = g1_data(candidate, tenant_id, cfg, pool_labels, used_labels)
    g2 = g2_performance(candidate, champion, cfg, segment_of)
    g3 = g3_calibration(candidate, champion, cfg)
    g4, g5 = g4_g5_trust(candidate, champion, txns_by_id, cfg)
    return GateReport([g1, g2, g3, g4, g5, g6_segments(g2),
                       Gate("G7", "Shadow", PENDING, "runs on live traffic after validation", {}),
                       g8_integrity(candidate, artefact_signed=artefact_signed, lineage_recorded=lineage_recorded),
                       Gate("G9", "Approval", PENDING, "needs a second person's sign-off", {})])
