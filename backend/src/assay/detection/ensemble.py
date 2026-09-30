"""Detection ensemble and calibration (PRD 3.1, FR-08, FR-09, FR-10).

Ensemble = bootstrap-resampled gradient-boosted trees + random forest + isolation forest.
Raw score is a weighted blend; the Fraud Risk Score is the raw score mapped to a probability with
isotonic regression fitted on a temporally later holdout. Prediction uncertainty is the spread of
the bootstrap members plus the distance from the operating threshold (FR-10).

The blend weights and all hyperparameters are parameters, not validated values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass(frozen=True)
class EnsembleConfig:
    seed: int = 0
    n_bootstrap: int = 5
    xgb_estimators: int = 120
    xgb_depth: int = 4
    xgb_learning_rate: float = 0.1
    rf_trees: int = 200
    rf_min_leaf: int = 5
    iso_trees: int = 100
    weights: tuple[float, float, float] = (0.55, 0.35, 0.10)  # boosted, forest, isolation


@dataclass
class Predictions:
    raw: np.ndarray
    calibrated: np.ndarray
    member_spread: np.ndarray
    distance_to_threshold: np.ndarray | None = None


@dataclass
class DetectionModel:
    config: EnsembleConfig
    feature_names: tuple[str, ...]
    boosters: list = field(default_factory=list)
    forest: RandomForestClassifier | None = None
    iso: IsolationForest | None = None
    iso_reference: np.ndarray | None = None  # sorted training scores, for percentile ranking
    baseline: LogisticRegression | None = None
    scaler: StandardScaler | None = None
    calibrator: IsotonicRegression | None = None

    # -- fitting ---------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> DetectionModel:
        cfg, rng = self.config, np.random.default_rng(self.config.seed)
        y = np.asarray(y).astype(int)
        self.boosters = []
        for i in range(cfg.n_bootstrap):
            idx = rng.integers(0, len(y), len(y))
            m = XGBClassifier(n_estimators=cfg.xgb_estimators, max_depth=cfg.xgb_depth,
                              learning_rate=cfg.xgb_learning_rate, subsample=0.9,
                              random_state=cfg.seed + i, n_jobs=2, verbosity=0)
            self.boosters.append(m.fit(X[idx], y[idx]))
        self.forest = RandomForestClassifier(
            n_estimators=cfg.rf_trees, min_samples_leaf=cfg.rf_min_leaf,
            class_weight="balanced_subsample", random_state=cfg.seed, n_jobs=2).fit(X, y)
        self.iso = IsolationForest(n_estimators=cfg.iso_trees, random_state=cfg.seed,
                                   n_jobs=2).fit(X)
        self.iso_reference = np.sort(-self.iso.score_samples(X))
        self.scaler = StandardScaler().fit(X)  # mandatory simple benchmark (PRD 3.1)
        self.baseline = LogisticRegression(max_iter=2000).fit(self.scaler.transform(X), y)
        # Training may run in parallel, but scoring must not: multi-threaded tree averaging sums
        # in a thread-dependent order, which changes results in the last bit and would break
        # exact replay (PRD 14.2, FR-33).
        self.forest.set_params(n_jobs=1)
        self.iso.set_params(n_jobs=1)
        for b in self.boosters:
            b.set_params(n_jobs=1)
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> DetectionModel:
        """Fit on a temporally LATER holdout than the training data (PRD 3.1)."""
        raw = self._raw(X_cal)[0]
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator.fit(raw, np.asarray(y_cal).astype(float))
        return self

    # -- scoring ---------------------------------------------------------------------
    def _iso_score(self, X: np.ndarray) -> np.ndarray:
        s = -self.iso.score_samples(X)
        return np.searchsorted(self.iso_reference, s, side="right") / len(self.iso_reference)

    def _raw(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        members = np.column_stack([b.predict_proba(X)[:, 1] for b in self.boosters])
        w_b, w_f, w_i = self.config.weights
        raw = (w_b * members.mean(axis=1) + w_f * self.forest.predict_proba(X)[:, 1]
               + w_i * self._iso_score(X))
        return raw, members

    def predict(self, X: np.ndarray, t_high: float | None = None) -> Predictions:
        if self.calibrator is None:
            raise RuntimeError("model is not calibrated; call calibrate() first")
        raw, members = self._raw(X)
        cal = self.calibrator.predict(raw)
        spread = members.std(axis=1)
        dist = None if t_high is None else np.abs(cal - t_high)
        return Predictions(raw, cal, spread, dist)

    def baseline_score(self, X: np.ndarray) -> np.ndarray:
        return self.baseline.predict_proba(self.scaler.transform(X))[:, 1]

    def params(self) -> dict:
        return asdict(self.config)
