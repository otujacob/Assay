"""Training pipeline: temporal split, fit, calibrate on a later holdout, evaluate on the future
(PRD 3.1, 6.3, 12.3 gate G1, FR-08, FR-09).

Split by event time into train < calibration < reliability < test. The reliability window is the
history that cohort reliability (Trust Index) is computed from; it is kept apart from calibration so
the model is judged there on data it was not calibrated on. Labels are matured outcomes only (labels.py).
Random splits are never used because they leak time.
"""

from __future__ import annotations

import hashlib
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from assay.features import (
    FEATURE_SET_VERSION,
    FeatureTable,
    build_table,
    definition_versions,
    find_leaks,
)
from assay.features.compute import parse_time
from assay.trust.reference import TrustReference, build_reference
from assay.trust.reliability import Cohort, CohortStore, amount_band, risk_band_name

from .bundle import BundleManifest
from .ensemble import DetectionModel, EnsembleConfig, Predictions
from .labels import labels_as_of
from .metrics import (
    cost_optimal_threshold,
    expected_calibration_error,
    pr_auc,
    precision_recall_at,
    reliability_curve,
)


@dataclass(frozen=True)
class TrainingConfig:
    tenant_id: str
    as_of: datetime  # "now": only outcomes matured by this time are used
    horizon_days: float  # a txn is labelled only once this old (>= dispute window + confirm delay)
    train_end: datetime
    calibration_end: datetime
    test_end: datetime
    reliability_end: datetime | None = None  # if set, [calibration_end, reliability_end) feeds cohort stats
    cost_fp: float = 1.0  # institution's cost matrix (PRD 10.4): relative, not a recommendation
    cost_fn: float = 20.0
    t_low_ratio: float = 0.33  # t_low = ratio * t_high; a parameter
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    leak_check_rows: int = 3000
    warmup_days: float = 30.0  # drop rows before history-window features reach steady state (0 = off)


@dataclass
class TrainingResult:
    model: DetectionModel
    manifest: BundleManifest
    test_table: FeatureTable
    test_y: np.ndarray
    test_pred: Predictions
    test_baseline: np.ndarray
    reference: TrustReference | None = None
    store: CohortStore | None = None
    train_table: FeatureTable | None = None

    def scoring_bundle(self) -> dict:
        """Everything needed to score and assess later: the artefact that gets signed and saved."""
        return {"model": self.model, "reference": self.reference, "store": self.store}


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "uncommitted"
    except (OSError, subprocess.SubprocessError):
        return "uncommitted"


def _rekey(store: CohortStore, bundle_id: str) -> CohortStore:
    out = CohortStore(store.edges)
    for c, (k, n) in list(store._stats.items()):
        out._stats[Cohort(c.product, c.channel, c.amount_band, c.risk_band, bundle_id)] = [k, n]
    return out


def dataset_id(txn_ids: list[str], y: np.ndarray) -> str:
    h = hashlib.sha256()
    for tid, label in sorted(zip(txn_ids, map(int, y), strict=True)):
        h.update(f"{tid}:{label};".encode())
    return h.hexdigest()


def train_bundle(txns: list[dict], outcomes: list[dict], cfg: TrainingConfig,
                 extra_labels: dict[str, int] | None = None) -> TrainingResult:
    """`extra_labels` (txn_id -> 0/1) are ACCEPTED analyst labels for cases with no matured verified outcome
    (learning/pool.py). They may only enter the TRAINING window. Calibration, reliability and test windows
    use verified outcomes only, so a candidate is never judged on labels that analysts influenced (PRD 11.4)."""
    # G1: point-in-time leakage check on a sample before anything is trained.
    leaks = find_leaks(sorted(txns, key=lambda x: x["event_time"])[: cfg.leak_check_rows],
                       sample_size=15)
    if leaks:
        raise AssertionError(f"point-in-time leakage detected: {leaks[:3]}")

    table = build_table(txns)
    labels = labels_as_of(txns, outcomes, cfg.as_of, cfg.horizon_days)
    extra = {t: int(v) for t, v in (extra_labels or {}).items() if t not in labels}
    by_time = {t["txn_id"]: parse_time(t["event_time"]) for t in txns} if extra else {}
    extra = {t: v for t, v in extra.items() if t in by_time and by_time[t] < cfg.train_end}
    labels = {**labels, **extra}
    keep = np.array([tid in labels for tid in table.txn_ids])
    if cfg.warmup_days:
        # History-window features (30-day velocity, beneficiary sharing) are still filling up at the
        # start of the data, so their distribution differs from steady state. Training on that
        # period made the drift monitor raise a false alarm on a clean test window (PRD 6.5).
        keep &= table.t >= table.t.min() + cfg.warmup_days * 86400
    table = table.subset(keep)
    y = np.array([labels[t] for t in table.txn_ids])

    def part(lo, hi):
        return (table.t >= (lo.timestamp() if lo else -np.inf)) & (table.t < hi.timestamp())

    tr, ca = part(None, cfg.train_end), part(cfg.train_end, cfg.calibration_end)
    rel = part(cfg.calibration_end, cfg.reliability_end) if cfg.reliability_end else None
    te = part(cfg.reliability_end or cfg.calibration_end, cfg.test_end)
    splits = [("train", tr), ("calibration", ca), ("test", te)] + ([("reliability", rel)] if rel is not None else [])
    for name, m in splits:
        if y[m].sum() < 5 or (1 - y[m]).sum() < 5:
            raise ValueError(f"{name} split has too few positives or negatives: "
                             f"{int(y[m].sum())}/{len(y[m])}")

    model = DetectionModel(cfg.ensemble, table.names).fit(table.X[tr], y[tr])
    model.calibrate(table.X[ca], y[ca])

    cal_pred = model.predict(table.X[ca])
    t_high = cost_optimal_threshold(y[ca], cal_pred.calibrated, cfg.cost_fp, cfg.cost_fn)
    t_low = cfg.t_low_ratio * t_high

    # Trust references: built from the training window and calibration predictions.
    reference = build_reference(table.X[tr], cal_pred.member_spread,
                                np.abs(cal_pred.calibrated - t_high), y_train=y[tr],
                                seed=cfg.ensemble.seed)
    store = CohortStore()
    if rel is not None:
        by_id = {t["txn_id"]: t for t in txns}
        rel_table, y_rel = table.subset(rel), y[rel]
        rel_pred = model.predict(rel_table.X, t_high)
        bid = "pending"  # the final bundle id is assigned below; cohorts are keyed by it after
        for i, tid in enumerate(rel_table.txn_ids):
            tx = by_id[tid]
            call = rel_pred.calibrated[i] >= t_high
            store.add(Cohort("default", str(tx.get("channel")), amount_band(float(tx["amount"]), store.edges),
                             risk_band_name(rel_pred.calibrated[i], t_low, t_high), bid),
                      bool(call) == bool(y_rel[i]))

    te_table = table.subset(te)
    y_te = y[te]
    pred = model.predict(te_table.X, t_high)
    base = model.baseline_score(te_table.X)
    metrics = {
        "test_rows": len(y_te), "test_positives": int(y_te.sum()),
        "test_base_rate": float(y_te.mean()),
        "pr_auc_ensemble": pr_auc(y_te, pred.calibrated),
        "pr_auc_logistic_baseline": pr_auc(y_te, base),
        "at_t_high": precision_recall_at(y_te, pred.calibrated, t_high),
    }
    calibration = {"ece_test": expected_calibration_error(y_te, pred.calibrated),
                   "reliability_test": reliability_curve(y_te, pred.calibrated),
                   "fit_on": "temporally later holdout than training"}
    manifest = BundleManifest(
        bundle_id=f"b-{uuid.uuid4().hex[:8]}", tenant_id=cfg.tenant_id,
        created_at=datetime.now(UTC).isoformat(), feature_set_version=FEATURE_SET_VERSION,
        definition_versions=definition_versions(), feature_names=list(table.names),
        dataset_id=dataset_id(table.txn_ids, y), dataset_tenant_ids=[cfg.tenant_id],
        code_commit=_git_commit(), params=model.params(),
        thresholds={"t_low": t_low, "t_high": t_high}, metrics=metrics, calibration=calibration,
        extra={"extra_labels_used": len(extra), "class_balance": {"train_pos": int(y[tr].sum()), "train_n": int(tr.sum()),
                                 "cal_pos": int(y[ca].sum()), "cal_n": int(ca.sum())},
               "cost_matrix": {"fp": cfg.cost_fp, "fn": cfg.cost_fn}})
    # Cohorts are keyed by model version; re-key from the provisional id to the real bundle id.
    store = _rekey(store, manifest.bundle_id)
    return TrainingResult(model, manifest, te_table, y_te, pred, base, reference, store,
                          table.subset(tr))
