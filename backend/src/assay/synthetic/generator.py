"""Synthetic data generator with known ground truth (PRD 28.3, Sprint 0).

Plants what the stress tests in PRD 6.5 need to find: known fraud types, one novel fraud type that
appears only late and looks ordinary, hard cases (camouflaged fraud, benign oddities), feature
drift, delayed outcomes, noisy analysts and degraded data. Output matches the TransactionEvent and
OutcomeEvent schemas of PRD 25.3.

Every signal a detector can use is realised in the events themselves (bursts are real
transactions, shared beneficiaries are real beneficiary ids), so features derived from history
see them. Hard cases exist on purpose: if fraud were trivially separable the detector would make
no errors and the Trust Index would have nothing to find.

Synthetic data proves the machinery works. It is not evidence that Assay works on real fraud
(PRD 28.3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

KNOWN_FRAUD_TYPES = ("card_testing", "account_takeover", "mule_ring")
NOVEL_FRAUD_TYPE = "novel_open_banking_scam"
CHANNELS = ("card_present", "card_not_present", "app", "web")
NOVEL_CHANNEL = "open_banking"
SCHEMA_VERSION = "txn-1"
START = datetime(2025, 1, 1, tzinfo=UTC)
MULE_BENEFICIARIES = tuple(f"b-mule-{i}" for i in range(5))
BURST_SIZE = (6, 12)


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 7
    tenant_id: str = "tenant-synth"
    n_days: int = 120
    txns_per_day: int = 400
    n_customers: int = 2000
    n_analysts: int = 8
    # Approximate per-transaction fraud rate. Card-testing bursts add extra fraud rows, so the
    # realised rate runs somewhat above this.
    base_fraud_rate: float = 0.02
    # Novel type only appears from this day (None disables it).
    novel_start_day: int | None = 90
    novel_share_of_fraud: float = 0.25
    # Hard cases
    fraud_camouflage: float = 0.25  # share of known fraud that looks like ordinary behaviour
    benign_anomaly_rate: float = 0.02  # share of legitimate txns that look like fraud
    # Feature drift: from drift_start_day, `amount_scale` multiplies amounts.
    drift_start_day: int | None = None
    amount_scale: float = 1.0
    # Labels
    dispute_window_days: int = 120  # OPD-2 working default
    fraud_confirm_delay_days: tuple[int, int] = (3, 40)
    outcome_coverage: float = 0.9  # share of fraud that ever gets a confirmed outcome
    # Analysts
    review_share: float = 0.15
    analyst_accuracy: float = 0.9
    bad_analyst_share: float = 0.0  # share of analysts who are near-random (noisy labels)
    blind_share: float = 0.1
    # Degraded data
    null_rate: float = 0.0


@dataclass
class SyntheticDataset:
    config: GeneratorConfig
    transactions: list[dict] = field(default_factory=list)
    outcomes: list[dict] = field(default_factory=list)
    analyst_actions: list[dict] = field(default_factory=list)
    truth: dict[str, dict] = field(default_factory=dict)  # txn_id -> hidden ground truth
    analyst_quality: dict[str, float] = field(default_factory=dict)

    def matured_outcomes(self, as_of: datetime) -> list[dict]:
        """Outcomes whose maturity time is on or before `as_of` (PRD 3.3)."""
        return [o for o in self.outcomes if datetime.fromisoformat(o["matured_at"]) <= as_of]


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def generate(cfg: GeneratorConfig | None = None) -> SyntheticDataset:
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    ds = SyntheticDataset(cfg)

    # Customers: a personal baseline amount and a home country.
    baseline = rng.lognormal(3.8, 0.6, cfg.n_customers)
    home = rng.choice(["GB", "IE", "FR", "DE"], cfg.n_customers, p=[0.85, 0.05, 0.05, 0.05])
    mule_customers = sorted(rng.choice(cfg.n_customers, 15, replace=False).tolist())

    analysts = [f"an-{i:02d}" for i in range(cfg.n_analysts)]
    n_bad = round(cfg.bad_analyst_share * cfg.n_analysts)
    for i, a in enumerate(analysts):
        ds.analyst_quality[a] = 0.55 if i < n_bad else cfg.analyst_accuracy

    # Card testing is emitted as a burst, so it is drawn less often per event.
    mean_burst = sum(BURST_SIZE) / 2
    type_p = np.array([1 / mean_burst, 1.0, 1.0])
    type_p = type_p / type_p.sum()

    counter = {"n": 0}

    def emit(t: datetime, cust: int, ftype: str | None, *, camouflage: bool, benign: bool,
             drifted: bool, burst_step: int = 0) -> None:
        counter["n"] += 1
        n = counter["n"]
        txn_id = f"t-{n:08d}"
        amount = float(baseline[cust] * rng.lognormal(0, 0.5))
        channel = str(rng.choice(CHANNELS))
        country = str(home[cust])
        new_device = bool(rng.random() < 0.05)
        beneficiary = f"b-{int(rng.integers(3000)):05d}" if rng.random() < 0.10 else None

        sig = ftype is not None and not camouflage
        if sig and ftype == "card_testing":
            amount = float(rng.uniform(0.5, 5))
            channel = "card_not_present"
        elif sig and ftype == "account_takeover":
            new_device = True
            beneficiary = f"b-{int(rng.integers(3000)):05d}"
            amount *= float(rng.uniform(3, 8))
            country = str(rng.choice(["NG", "RO", "BR"]))
        elif sig and ftype == "mule_ring":
            amount = float(rng.uniform(900, 2500))
            beneficiary = str(rng.choice(MULE_BENEFICIARIES))
        elif ftype == NOVEL_FRAUD_TYPE:
            # Looks ordinary except for an unseen channel and a fresh beneficiary.
            channel = NOVEL_CHANNEL
            amount = float(baseline[cust] * rng.uniform(1.0, 2.5))
            beneficiary = f"b-{int(rng.integers(3000)):05d}"
        elif benign:  # legitimate but unusual: travel, a big purchase
            new_device = True
            amount *= float(rng.uniform(3, 6))
            country = str(rng.choice(["US", "ES", "AE"]))

        if drifted:
            amount *= cfg.amount_scale

        txn = {
            "event_id": f"e-txn-{n:08d}",
            "txn_id": txn_id,
            "event_time": _iso(t),
            "amount": round(amount, 2),
            "currency": "GBP",
            "channel": channel,
            "customer_pid": f"c-{cust:05d}",
            "account_pid": f"a-{cust:05d}",
            "beneficiary_pid": beneficiary,
            "merchant_id": f"m-{int(rng.integers(500)):04d}",
            "device_hash": f"d-{cust:05d}-{'new' if new_device else 'known'}",
            "ip_hash": f"ip-{int(rng.integers(10**6)):06d}",
            "country": country,
            "schema_version": SCHEMA_VERSION,
        }
        # Generator-internal context, stripped by to_wire().
        txn["_ctx"] = {"baseline_amount": round(float(baseline[cust]), 2),
                       "home_country": str(home[cust]), "new_device": new_device,
                       "camouflaged": bool(ftype and camouflage), "benign_anomaly": benign}

        if cfg.null_rate > 0:
            for k in ("device_hash", "ip_hash", "country", "merchant_id"):
                if rng.random() < cfg.null_rate:
                    txn[k] = None

        ds.transactions.append(txn)
        ds.truth[txn_id] = {"is_fraud": ftype is not None, "fraud_type": ftype}

        # Outcomes: fraud confirmed after a delay, legit confirmed at window end.
        if ftype is not None:
            if rng.random() < cfg.outcome_coverage:
                lo, hi = cfg.fraud_confirm_delay_days
                ev = t + timedelta(days=int(rng.integers(lo, hi + 1)))
                ds.outcomes.append(_outcome(len(ds.outcomes), txn_id, "confirmed_fraud",
                                            "chargeback", ev, ev))
        else:
            ev = t + timedelta(days=cfg.dispute_window_days)
            ds.outcomes.append(_outcome(len(ds.outcomes), txn_id, "confirmed_legitimate",
                                        "dispute_window_closed", ev, ev))

        # Analyst review of a random share; blind flag; accuracy per analyst.
        if rng.random() < cfg.review_share:
            a = str(rng.choice(analysts))
            correct = rng.random() < ds.analyst_quality[a]
            says_fraud = (ftype is not None) if correct else (ftype is None)
            ds.analyst_actions.append({
                "action_id": f"aa-{len(ds.analyst_actions):07d}",
                "txn_id": txn_id,
                "analyst_pid": a,
                "action": "block" if says_fraud else "approve",
                "blind_flag": bool(rng.random() < cfg.blind_share),
                "action_time": _iso(t + timedelta(hours=int(rng.integers(1, 30)))),
            })

    for day in range(cfg.n_days):
        count = int(rng.poisson(cfg.txns_per_day))
        drifted = cfg.drift_start_day is not None and day >= cfg.drift_start_day
        novel_on = cfg.novel_start_day is not None and day >= cfg.novel_start_day
        for _ in range(count):
            cust = int(rng.integers(cfg.n_customers))
            t = START + timedelta(days=day, seconds=int(rng.integers(86400)))

            ftype = None
            if rng.random() < cfg.base_fraud_rate:
                if novel_on and rng.random() < cfg.novel_share_of_fraud:
                    ftype = NOVEL_FRAUD_TYPE
                else:
                    ftype = str(rng.choice(KNOWN_FRAUD_TYPES, p=type_p))
            camouflage = (ftype in KNOWN_FRAUD_TYPES) and rng.random() < cfg.fraud_camouflage
            benign = ftype is None and rng.random() < cfg.benign_anomaly_rate

            if ftype == "mule_ring":
                cust = int(rng.choice(mule_customers))
            if ftype == "card_testing" and not camouflage:
                k = int(rng.integers(BURST_SIZE[0], BURST_SIZE[1] + 1))
                for step in range(k):  # a real burst: seconds apart on one card
                    emit(t + timedelta(seconds=step * int(rng.integers(5, 40))), cust, ftype,
                         camouflage=False, benign=False, drifted=drifted, burst_step=step)
            else:
                emit(t, cust, ftype, camouflage=camouflage, benign=benign, drifted=drifted)
    return ds


def _outcome(i: int, txn_id: str, otype: str, source: str, event_time: datetime,
             matured_at: datetime) -> dict:
    return {
        "event_id": f"e-out-{i:08d}",
        "txn_id": txn_id,
        "outcome_type": otype,
        "source": source,
        "event_time": _iso(event_time),
        "matured_at": _iso(matured_at),
        "schema_version": "out-1",
    }


def to_wire(row: dict) -> dict:
    """Strip generator-internal fields so a row matches the PRD 25.3 event schema."""
    return {k: v for k, v in row.items() if k not in ("_ctx", "matured_at")}


def write_jsonl(ds: SyntheticDataset, out_dir: str | Path) -> None:
    """Write events to `out_dir`. Ground truth goes in a separate file, never in the events."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("transactions", ds.transactions), ("outcomes", ds.outcomes),
                       ("analyst_actions", ds.analyst_actions)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(to_wire(r) if name != "analyst_actions" else r) + "\n")
    (out / "truth.json").write_text(json.dumps(ds.truth), encoding="utf-8")
