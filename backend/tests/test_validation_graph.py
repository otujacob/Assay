"""The helpers behind the H6 experiment (validation/graph.py): subset metrics and the false-link count."""

from datetime import UTC, datetime, timedelta

import numpy as np

from assay.validation.graph import false_links, subset_pr_auc

T0 = datetime(2025, 1, 1, tzinfo=UTC)


def txn(i, cust, day, **kw):
    return {"txn_id": f"t{i}", "customer_pid": cust, "account_pid": f"a-{cust}", "event_time": (T0 + timedelta(days=day)).isoformat(),
            "device_hash": kw.get("device"), "ip_hash": kw.get("ip"), "beneficiary_pid": kw.get("ben"), "merchant_id": None}


def test_subset_pr_auc_judges_ring_cases_and_other_fraud_separately():
    ids = ["a", "b", "c", "d", "e", "f"]
    truth = {"a": {"fraud_type": "ring_attack"}, "b": {"fraud_type": "account_takeover"}, "c": {"fraud_type": None},
             "d": {"fraud_type": None}, "e": {"fraud_type": None}, "f": {"fraud_type": None}}
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.5, 0.4, 0.3, 0.2])  # ranks the ring fraud first and the other fraud last
    ring = subset_pr_auc(y, p, ids, truth, ring=True)
    other = subset_pr_auc(y, p, ids, truth, ring=False)
    assert ring == 1.0 and other < 0.5
    assert np.isnan(subset_pr_auc(np.array([0, 0, 0, 0, 0, 0]), p, ids, truth, ring=True))  # no positives to judge


def test_false_links_counts_links_inside_planted_groups_as_true_and_the_rest_as_false():
    txns = []
    n = 0
    for day in range(4):  # a ring shares a beneficiary; a family shares a device; two strangers share a device by chance
        for c in ("r1", "r2", "r3"):
            txns.append(txn(n := n + 1, c, day, ben="b-ring"))
        for c in ("f1", "f2"):
            txns.append(txn(n := n + 1, c, day, device="d-fam"))
        for c in ("s1", "s2"):
            txns.append(txn(n := n + 1, c, day, device="d-chance"))
    structure = {"rings": {"ring-0": ["r1", "r2", "r3"]}, "families": {"family-0": ["f1", "f2"]}}
    out = false_links(txns, structure, (T0 + timedelta(days=4)).timestamp())
    assert out["true"] == 4 and out["false"] == 1  # 3 ring pairs + 1 family pair; the chance pair is false
    assert out["false_share"] == 1 / 5
    assert out["by_relationship"]["SHARES_DEVICE"] == {"true": 1, "false": 1}
    assert out["by_relationship"]["SHARES_BENEFICIARY"] == {"true": 3, "false": 0}


def test_false_links_with_no_links_has_no_share():
    out = false_links([txn(1, "a", 0, device="d1"), txn(2, "b", 0, device="d2")], {}, (T0 + timedelta(days=1)).timestamp())
    assert out["links"] == 0 and out["false_share"] is None
