"""Trust assessor: turns predictions into one TrustAssessment per case (PRD 5, FR-14 to FR-20).

Inputs are a feature table, the model's predictions, and a TrustReference. Components: conf, rel,
exp, fam, drift, dq computed here; hum is inactive in the MVP (PRD 5.4), so weights renormalise
over the rest. A component that cannot be computed for a case is MISSING and penalised.

OPD-7 (which cases get explanation testing) is the `explain_mask`: cases outside it have exp
missing. With exp missing, TI_low is capped near 48 (PRD 5.4), so such cases can never be High.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from assay.detection.ensemble import DetectionModel, Predictions
from assay.features.registry import CHANNEL_ORDER  # noqa: F401  (documented dependency)

from .confidence import model_confidence
from .drift import drift_exposure
from .explain import ExplainConfig, ExplanationResult, attributions, explain_batch
from .index import (
    Component,
    TrustConfig,
    TrustContext,
    TrustResult,
    compute_trust_index,
)
from .quality import QualityConfig, data_quality
from .reference import TrustReference
from .reliability import Cohort, CohortStore, amount_band, risk_band_name


@dataclass
class CaseAssessment:
    txn_id: str
    result: TrustResult
    components: dict[str, Component]
    novelty: float
    drift_exposure: float
    data_quality: dict[str, float]
    cohort: Cohort
    explanation: dict = field(default_factory=dict)  # sub-scores, for storage
    provisional: TrustResult | None = None  # the fixed-weight result, kept when calibrated mode replaced `result`


class TrustAssessor:
    def __init__(self, model: DetectionModel, reference: TrustReference, store: CohortStore,
                 thresholds: dict[str, float], *, model_version: str,
                 trust_cfg: TrustConfig | None = None, explain_cfg: ExplainConfig | None = None,
                 quality_cfg: QualityConfig | None = None, product: str = "default", calibrated=None):
        """`calibrated` is an enabled CalibratedTrustModel (assay.trust.calibrated), or None for Provisional
        mode. It is duck-typed here (`enabled`, `predict`, `apply`) because that module imports this one."""
        if calibrated is not None and not calibrated.enabled:
            raise ValueError("a calibrated Trust Index can only be used after it passed the section 6 gate")
        self.calibrated = calibrated
        self.model, self.ref, self.store = model, reference, store
        self.t_low, self.t_high = thresholds["t_low"], thresholds["t_high"]
        self.model_version, self.product = model_version, product
        self.trust_cfg = trust_cfg or TrustConfig()
        self.explain_cfg = explain_cfg or ExplainConfig()
        self.quality_cfg = quality_cfg

    def assess(self, X: np.ndarray, txns: list[dict], preds: Predictions, *,
               drift_vector: np.ndarray | None = None, explain_mask: np.ndarray | None = None,
               source_health: float = 1.0, recorded_at: list | None = None):
        """Assess every row. `drift_vector` is the current per-feature drift (None = no drift
        information, treated as no drift). `explain_mask` selects cases given explanation testing."""
        n = len(txns)
        ids = [t["txn_id"] for t in txns]
        mask = np.ones(n, bool) if explain_mask is None else np.asarray(explain_mask, bool)

        conf, conf_lo, conf_hi = model_confidence(
            preds.member_spread, np.abs(preds.calibrated - self.t_high),
            self.ref.spread_ecdf, self.ref.distance_ecdf)
        novelty = self.ref.novelty(X)

        # Explanations only where the compute policy says so.
        expl: ExplanationResult | None = None
        if mask.any():
            expl = explain_batch(self.model.boosters, X[mask], [ids[i] for i in np.flatnonzero(mask)],
                                 self.ref, self.explain_cfg)
        pos = {int(g): k for k, g in enumerate(np.flatnonzero(mask))}
        # Attributions are cheap (one TreeSHAP pass) and are all that drift exposure needs (PRD 9.3),
        # so drift is available for EVERY case. Only the perturbation tests behind exp are costly.
        # Without this, skipping explanation testing also dropped drift, and two missing components
        # widened the Trust Index interval enough to push most cases to Insufficient evidence.
        attrs = attributions(self.model.boosters, X)
        dvec = drift_vector if drift_vector is not None else np.zeros(X.shape[1])
        drift_D = drift_exposure(attrs, dvec)

        out: list[CaseAssessment] = []
        for i, t in enumerate(txns):
            risk_band = risk_band_name(preds.calibrated[i], self.t_low, self.t_high)
            cohort = Cohort(self.product, str(t.get("channel")), amount_band(float(t["amount"]),
                            self.store.edges), risk_band, self.model_version)
            rel, rel_lo, rel_hi, n_rel = self.store.reliability(cohort)
            q = data_quality(t, recorded_at=None if recorded_at is None else recorded_at[i],
                             source_health=source_health, cfg=self.quality_cfg)

            comps = {
                "conf": Component.active(float(conf[i]), float(conf_lo[i]), float(conf_hi[i]), 1),
                "rel": Component.active(rel, rel_lo, rel_hi, n_rel),
                "fam": Component.active(float(1.0 - novelty[i]), n=1),
                "dq": Component.active(q["q"], n=1),
                "hum": Component.inactive(),
            }
            ex = {}
            if i in pos and expl is not None:
                k = pos[i]
                e = expl.exp[k]
                if np.isnan(e):
                    comps["exp"] = Component.missing()
                else:
                    comps["exp"] = Component.active(float(e), float(expl.exp_lo[k]),
                                                    float(expl.exp_hi[k]),
                                                    int(expl.n_valid_perturbations[k]))
                ex = {"stability": float(expl.stability[k]), "sensitivity": float(expl.sensitivity[k]),
                      "faithfulness": float(expl.faithfulness[k]),
                      "reproducible": bool(expl.reproducible[k]),
                      "group_attributions": dict(zip(expl.group_names, expl.group_attr[k].tolist(),
                                                     strict=True))}
            else:
                comps["exp"] = Component.missing()
            D = float(drift_D[i])
            comps["drift"] = Component.active(1.0 - D, n=1)

            ctx = TrustContext(model_matured_outcomes=self.store.total())
            result = compute_trust_index(comps, self.trust_cfg, ctx)
            out.append(CaseAssessment(ids[i], result, comps, float(novelty[i]), D, q, cohort, ex))
        if self.calibrated is not None:
            # Calibrated mode (PRD 5.6). The Insufficient-evidence gates have already run above and are not
            # reopened: a case with no provisional score keeps its reason codes and gets no calibrated one.
            scored = self.calibrated.predict(out)
            for case, p in zip(out, scored, strict=True):
                case.provisional = case.result
                case.result = self.calibrated.apply(case, p)
        return out
