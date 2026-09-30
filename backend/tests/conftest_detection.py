"""Shared fixtures for detection tests (imported by test modules, not auto-collected)."""

from dataclasses import replace
from datetime import timedelta

from assay.detection import TrainingConfig
from assay.synthetic import START, GeneratorConfig, generate, to_wire

GEN = GeneratorConfig(seed=5, n_days=240, txns_per_day=110, n_customers=500, novel_start_day=170,
                      dispute_window_days=30, fraud_confirm_delay_days=(3, 40),
                      base_fraud_rate=0.03)


def make_dataset(seed: int | None = None):
    gen = GEN if seed is None else replace(GEN, seed=seed)
    ds = generate(gen)
    txns = [to_wire(t) for t in ds.transactions]
    cfg = TrainingConfig(
        tenant_id="tenant-synth", as_of=START + timedelta(days=GEN.n_days), horizon_days=40,
        train_end=START + timedelta(days=100), calibration_end=START + timedelta(days=130),
        reliability_end=START + timedelta(days=165), test_end=START + timedelta(days=200))
    return ds, txns, cfg
