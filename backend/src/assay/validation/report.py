"""Validation report storage (FR-39) and a readable rendering.

Reports are stored append-only and linked to the bundle and dataset they evaluated, so the
evidence for enabling Calibrated mode or setting thresholds can be audited.
"""

from __future__ import annotations

from assay.detection.bundle import BundleManifest
from assay.scoring.service import ensure_bundle_row

from .harness import _jsonable

SCHEMA = "val-1"


def store_report(repo, tenant_id: str, manifest: BundleManifest, report: dict,
                 stress: dict | None = None, actor: str = "validation") -> dict:
    ensure_bundle_row(repo, tenant_id, manifest, actor)
    return repo.insert(tenant_id, "validation_reports", {
        "schema_version": SCHEMA, "bundle_id": manifest.bundle_id,
        "dataset_id": report["meta"]["dataset_id"], "mode": report["meta"]["mode"],
        "n_cases": report["meta"]["n_cases"],
        "measures": _jsonable(report["measures"]), "baselines": _jsonable(report["baselines"]),
        "ablations": _jsonable(report["ablations"]),
        "stress_results": None if stress is None else _jsonable(stress),
        "pass_criteria": _jsonable(report["pass_criteria"]),
        "limitations": report["limitations"]}, actor)


def _pct(x, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _rate(r: dict) -> str:
    if r is None or r.get("value") is None:
        return "n/a (no cases)"
    return f"{_pct(r['value'], 2)} (95% CI {_pct(r['lo'], 2)} to {_pct(r['hi'], 2)}; {r['k']}/{r['n']})"


def _auc(a: dict) -> str:
    return f"{a['value']:.3f} (95% CI {a['lo']:.3f} to {a['hi']:.3f})" if a["lo"] is not None else f"{a['value']:.3f}"


def render_markdown(report: dict, stress: dict | None = None) -> str:
    m, meta, pc = report["measures"], report["meta"], report["pass_criteria"]
    verdict = pc["atce_supported_as_specified"]
    if verdict is True:
        verdict_text = "**Verdict against the PRD 6.6 criteria: SUPPORTED.**"
    elif verdict is False:
        verdict_text = ("**Verdict against the PRD 6.6 criteria: NOT SUPPORTED as specified.** "
                        + pc["if_not_supported"] + ".")
    else:
        verdict_text = "**Verdict against the PRD 6.6 criteria: INCONCLUSIVE** (too few high-trust cases)."
    lines = [
        f"# Trust Index validation report: bundle {meta['bundle_id']}",
        "",
        (f"Mode: **{meta['mode']}**. Cases: {meta['n_cases']} ({meta['n_wrong']} wrong recommendations, "
         f"{_pct(meta['n_wrong'] / meta['n_cases'], 2)}). Dataset {meta['dataset_id'][:12]}. "
         f"Generated {meta['created_at'][:19]}Z."),
        "",
        verdict_text,
        "",
        "## Headline measures (matured verified outcomes, later than all training data)",
        f"- Error discrimination, AUROC of (100 - TI): {_auc(m['auroc'])}",
        (f"- High-trust error rate: {_rate(m['high_trust_error_rate'])} against overall error rate "
         f"{_rate(report['overall_error_rate'])}"),
        f"- Low-trust detection rate (share of errors sent to a human): {_rate(m['low_trust_detection_rate'])}",
        f"- False-confidence rate (share of errors that had High trust): {_rate(m['false_confidence_rate'])}",
        f"- Coverage (cases that received a score): {_pct(m['scored_share'])}",
        f"- Trust states: {m['state_counts']}",
        "",
        "## Against simple baselines (PRD 6.4)",
        "| Baseline | AUROC | Trust Index minus baseline | Beats it? |",
        "|---|---|---|---|",
    ]
    for k in ("B1_distance_from_threshold", "B2_ensemble_disagreement", "B3_max_class_probability"):
        b = report["baselines"][k]
        d = b["trust_index_minus_baseline"]
        beats = pc["trust_index_beats_B1_to_B3_with_ci_excluding_no_improvement"][k]
        lines.append(f"| {k} | {_auc(b['auroc'])} | {d['diff']:+.3f} (CI {d['lo']:+.3f} to {d['hi']:+.3f}) "
                     f"| {'yes' if beats else 'no (interval includes no improvement)'} |")
    lines += ["", "Each component alone (B4), AUROC: " + ", ".join(
        f"{c} {v['auroc']:.3f}" for c, v in report["baselines"]["B4_single_components"].items()),
        "", "## Ablation: remove each component in turn", "| Component | AUROC without it | Change | Verdict |",
        "|---|---|---|---|"]
    for c, a in report["ablations"].items():
        lines.append(f"| {c} | {a['auroc']:.3f} | {a['delta_auroc']:+.3f} | {a['verdict']} |")
    if stress:
        lines += ["", "## Stress tests (PRD 6.5)"]
        for s in stress.values():
            if s.get("status") == "not_testable":
                st = "not testable"
            elif s.get("passed") is None:
                st = "reported, no pass rule"
            else:
                st = "passed" if s["passed"] else "FAILED"
            lines.append(f"- **{s['test']}**: {st}. Expected: {s['expected']}.")
    lines += ["", "## Limitations"] + [f"- {x}" for x in report["limitations"]]
    return "\n".join(lines) + "\n"
