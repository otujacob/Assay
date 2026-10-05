"""Does the entity graph add detection value over the same entity data used as flat features? (PRD 20, H6)

Three arms are trained identically and scored on the same verified-outcome-only test window:
  F0  the registry's flat features (what the production model uses)
  F1  F0 plus the flat entity features of graph/features.py (plain 30-day counts per entity, and how many of those
      customers have a confirmed fraud known at the time): the same entity data, no graph
  G   F0 plus the graph features (confidence-weighted, decayed, two-hop, communities)

This repeats the first steps of detection/train.py (temporal split, fit, calibrate on a later window) so that the
extra columns can be added without changing the production training path. Nothing here is used by scoring: the graph
stays investigator context unless this question is answered with a margin (PRD 20, H6).
"""

from __future__ import annotations

import numpy as np

from assay.detection import TrainingConfig
from assay.detection.ensemble import DetectionModel
from assay.detection.labels import labels_as_of
from assay.detection.metrics import pr_auc
from assay.features import build_table
from assay.graph.features import FLAT_FEATURES, GRAPH_FEATURES, build_graph_table
from assay.graph.temporal import GraphConfig, TemporalGraph


def prepare(txns: list[dict], outcomes: list[dict], cfg: TrainingConfig, graph_cfg: GraphConfig | None = None) -> dict:
    """Everything the arms share: the flat table, its labels and splits, and the graph and flat-entity features."""
    table = build_table(txns)
    labels = labels_as_of(txns, outcomes, cfg.as_of, cfg.horizon_days)
    keep = np.array([t in labels for t in table.txn_ids])
    if cfg.warmup_days:
        keep &= table.t >= table.t.min() + cfg.warmup_days * 86400
    ids, G, F = build_graph_table(txns, outcomes, graph_cfg)
    pos = {t: i for i, t in enumerate(ids)}
    assert set(table.txn_ids) <= set(pos), "the graph table must cover every transaction"
    order = np.array([pos[t] for t in table.txn_ids])
    table = table.subset(keep)
    order = order[keep]
    y = np.array([labels[t] for t in table.txn_ids])

    def part(lo, hi):
        return (table.t >= (lo.timestamp() if lo else -np.inf)) & (table.t < hi.timestamp())

    splits = {"train": part(None, cfg.train_end), "cal": part(cfg.train_end, cfg.calibration_end),
              "test": part(cfg.reliability_end or cfg.calibration_end, cfg.test_end)}
    return {"table": table, "y": y, "G": G[order], "F": F[order], "splits": splits}


def fit_arm(prep: dict, cfg: TrainingConfig, extra: np.ndarray | None, extra_names: tuple[str, ...]) -> dict:
    t, y, sp = prep["table"], prep["y"], prep["splits"]
    X = t.X if extra is None else np.hstack([t.X, extra])
    names = list(t.names) + list(extra_names)
    model = DetectionModel(cfg.ensemble, names).fit(X[sp["train"]], y[sp["train"]])
    model.calibrate(X[sp["cal"]], y[sp["cal"]])
    te = sp["test"]
    return {"p": model.predict(X[te]).calibrated, "y": y[te], "ids": [i for i, k in zip(t.txn_ids, te, strict=True) if k]}


def arms(prep: dict, cfg: TrainingConfig) -> dict[str, dict]:
    return {"F0": fit_arm(prep, cfg, None, ()),
            "F1": fit_arm(prep, cfg, prep["F"], FLAT_FEATURES),
            "G": fit_arm(prep, cfg, prep["G"], GRAPH_FEATURES)}


def subset_pr_auc(y: np.ndarray, p: np.ndarray, ids: list[str], truth: dict[str, dict], *, ring: bool) -> float:
    """PR-AUC on legitimate rows plus only the ring frauds (ring=True) or only the other frauds (ring=False)."""
    is_ring = np.array([truth[i]["fraud_type"] == "ring_attack" for i in ids])
    keep = (y == 0) | (is_ring if ring else ~is_ring)
    if y[keep].sum() == 0:
        return float("nan")
    return pr_auc(y[keep], p[keep])


def false_links(graph_txns: list[dict], structure: dict, t: float, cfg: GraphConfig | None = None) -> dict:
    """Descriptive: of the derived links the graph holds at time `t` above its threshold, how many join customers who are
    in the same planted ring or family (true structure) and how many join unrelated customers (false links)?"""
    from assay.features.compute import parse_time

    g = TemporalGraph(cfg)
    for x in graph_txns:
        if parse_time(x["event_time"]).timestamp() <= t:
            g.observe(x)
    same = {}
    for kind in ("rings", "families"):
        for gid, members in structure.get(kind, {}).items():
            for m in members:
                same[m] = gid
    custs = sorted({x["customer_pid"] for x in graph_txns})
    seen: set[tuple[str, str]] = set()
    true = false = 0
    by_rel: dict[str, list[int]] = {}
    for c in custs:
        for lk in g.links(c, t):
            if lk.confidence < g.cfg.link_threshold:
                continue
            key = tuple(sorted((c, lk.other)))
            if key in seen:
                continue
            seen.add(key)
            ok = c in same and same.get(c) == same.get(lk.other)
            true += ok
            false += not ok
            by_rel.setdefault(lk.rel, [0, 0])[0 if ok else 1] += 1
    return {"links": true + false, "true": true, "false": false,
            "false_share": (false / (true + false)) if true + false else None,
            "by_relationship": {k: {"true": v[0], "false": v[1]} for k, v in by_rel.items()}}
