import { useState } from "react";
import type { BundleInfo, ValidationReport } from "../types";
import { num, pct } from "../lib/format";
import { Pill } from "./Pill";

type Tab = "validation" | "performance" | "bundle";

function ReliabilityCurve({ points }: { points: { mean_predicted: number; observed_rate: number }[] }) {
  const max = Math.max(0.05, ...points.flatMap((p) => [p.mean_predicted, p.observed_rate])) * 1.1;
  const W = 240, H = 170, L = 34, B = 24, T = 8, Rr = 8;
  const x = (v: number) => L + (v / max) * (W - L - Rr);
  const y = (v: number) => H - B - (v / max) * (H - B - T);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Fraud risk calibration reliability curve">
      <line x1={L} y1={H - B} x2={W - Rr} y2={H - B} className="axis" />
      <line x1={L} y1={T} x2={L} y2={H - B} className="axis" />
      <line x1={x(0)} y1={y(0)} x2={x(max)} y2={y(max)} className="diag" />
      <polyline className="curve" points={points.map((p) => `${x(p.mean_predicted)},${y(p.observed_rate)}`).join(" ")} />
      {points.map((p, i) => <circle key={i} cx={x(p.mean_predicted)} cy={y(p.observed_rate)} r={2.6} className="dot" />)}
      <text x={(L + W) / 2} y={H - 4} className="axis-label" textAnchor="middle">Predicted risk</text>
      <text x={8} y={H / 2} className="axis-label" transform={`rotate(-90 8 ${H / 2})`} textAnchor="middle">Observed rate</text>
    </svg>
  );
}

function BaselineBars({ report }: { report: ValidationReport }) {
  const rows = [
    { label: "Trust Index", v: report.measures.auroc, key: "ti" },
    { label: "B1 Threshold distance", v: report.baselines.B1_distance_from_threshold.auroc, key: "b1" },
    { label: "B2 Disagreement", v: report.baselines.B2_ensemble_disagreement.auroc, key: "b2" },
    { label: "B3 Max probability", v: report.baselines.B3_max_class_probability.auroc, key: "b3" },
  ];
  return (
    <div data-testid="baseline-bars">
      {rows.map((r) => (
        <div className="bl-row" key={r.key}>
          <div className="bl-label small">{r.label}</div>
          <div className="bl-track">
            <div className={`bl-fill ${r.key === "ti" ? "ti" : ""}`} style={{ width: `${r.v.value * 100}%` }} />
            {r.v.lo !== null && r.v.hi !== null && (
              <div className="bl-ci" style={{ left: `${r.v.lo * 100}%`, width: `${(r.v.hi - r.v.lo) * 100}%` }} />
            )}
          </div>
          <div className="bl-val mono small">{num(r.v.value, 3)}</div>
        </div>
      ))}
      <div className="tiny muted pad-top">AUROC for ranking wrong recommendations (higher is better), with 95% intervals.</div>
    </div>
  );
}

export function Governance({ report, bundle }: { report: ValidationReport | null; bundle: BundleInfo | null }) {
  const [tab, setTab] = useState<Tab>("validation");
  const tabs: { id: Tab; label: string }[] = [
    { id: "validation", label: "Validation & Governance" },
    { id: "performance", label: "Performance & Calibration" },
    { id: "bundle", label: "Model Bundle & Lineage" },
  ];
  const verdict = report?.pass_criteria.atce_supported_as_specified;
  if (!report && !bundle) {
    return (
      <div className="panel" data-testid="governance">
        <div className="empty pad">Governance evidence (validation, calibration, model bundle) is available to the auditor, model-risk approver and manager roles.</div>
      </div>
    );
  }
  return (
    <div className="panel" data-testid="governance">
      <div className="tabs" role="tablist">
        {tabs.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className={`tab ${tab === t.id ? "on" : ""}`} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="gov-note tiny">
        <Pill tone="amber">Synthetic data</Pill> These figures come from a validation run on synthetic data. They show the
        machinery works; they are not evidence about real fraud.
      </div>

      {tab === "validation" && (
        <div className="gov-grid">
          <div className="card">
            <div className="card-title">1. Fraud risk calibration (held-out period)</div>
            {bundle?.calibration?.reliability_test ? (
              <>
                <ReliabilityCurve points={bundle.calibration.reliability_test} />
                <div className="tiny muted">ECE {num(bundle.calibration.ece_test, 4)} · applies to the Fraud Risk Score, not the Trust Index (Provisional mode makes no probability claim).</div>
              </>
            ) : <div className="muted small">No calibration report.</div>}
          </div>
          <div className="card">
            <div className="card-title">2. Baseline comparison</div>
            {report ? <BaselineBars report={report} /> : <div className="muted small">No validation report stored.</div>}
          </div>
          <div className="card">
            <div className="card-title">3. Component ablation</div>
            {report ? (
              <table className="mini" data-testid="ablation-table">
                <thead><tr><th>Component</th><th className="right">AUROC without</th><th className="right">Change</th><th>Reading</th></tr></thead>
                <tbody>
                  {Object.entries(report.ablations).map(([c, a]) => (
                    <tr key={c}>
                      <td>{c}</td><td className="right mono">{num(a.auroc, 3)}</td>
                      <td className="right mono">{a.delta_auroc >= 0 ? "+" : ""}{num(a.delta_auroc, 3)}</td>
                      <td className="tiny">{a.verdict}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <div className="muted small">No validation report stored.</div>}
          </div>
          <div className="card">
            <div className="card-title">4. Verdict (PRD 6.6)</div>
            {report ? (
              <>
                <Pill tone={verdict === true ? "green" : verdict === false ? "red" : "grey"}>
                  {verdict === true ? "Supported" : verdict === false ? "Not supported as specified" : "Inconclusive"}
                </Pill>
                <div className="small pad-top">
                  {Object.entries(report.pass_criteria.trust_index_beats_B1_to_B3_with_ci_excluding_no_improvement).map(([k, v]) => (
                    <div key={k}>{v ? "✓" : "✗"} beats {k.split("_")[0]} with interval excluding no improvement</div>
                  ))}
                  <div>{report.pass_criteria.high_trust_error_rate_materially_below_overall ? "✓" : "✗"} high-trust error rate below overall</div>
                </div>
                <div className="tiny muted pad-top">The criteria are not relaxed when they fail. See docs/validation/README.md.</div>
              </>
            ) : <div className="muted small">No validation report stored.</div>}
          </div>
        </div>
      )}

      {tab === "performance" && report && (
        <div className="gov-grid">
          <div className="card">
            <div className="card-title">Trust measures ({report.n_cases} cases)</div>
            <table className="mini">
              <tbody>
                <tr><td>Error discrimination (AUROC)</td><td className="mono right">{num(report.measures.auroc.value, 3)}</td></tr>
                <tr><td>High-trust error rate</td><td className="mono right">{pct(report.measures.high_trust_error_rate.value, 2)}</td></tr>
                <tr><td>Low-trust detection rate</td><td className="mono right">{pct(report.measures.low_trust_detection_rate.value)}</td></tr>
                <tr><td>False-confidence rate</td><td className="mono right">{pct(report.measures.false_confidence_rate.value)}</td></tr>
                <tr><td>Coverage (scored)</td><td className="mono right">{pct(report.measures.scored_share)}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="card">
            <div className="card-title">Stress tests (PRD 6.5)</div>
            {report.stress_results ? Object.values(report.stress_results).map((s) => (
              <div className="small hist-row" key={s.test}>
                <Pill tone={s.status === "not_testable" ? "grey" : s.passed ? "green" : s.passed === false ? "red" : "grey"}>
                  {s.status === "not_testable" ? "not testable" : s.passed === null ? "reported" : s.passed ? "passed" : "failed"}
                </Pill> {s.test}
              </div>
            )) : <div className="muted small">Not run.</div>}
          </div>
          <div className="card">
            <div className="card-title">Limitations</div>
            <ul className="limits small">{report.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
          </div>
        </div>
      )}
      {tab === "performance" && !report && <div className="empty pad">No validation report stored.</div>}

      {tab === "bundle" && bundle && (
        <div className="gov-grid">
          <BundleCard bundle={bundle} />
          <div className="card">
            <div className="card-title">Operating thresholds</div>
            <table className="mini"><tbody>
              <tr><td>t_high (fraud call)</td><td className="mono right">{num(bundle.thresholds?.t_high, 4)}</td></tr>
              <tr><td>t_low</td><td className="mono right">{num(bundle.thresholds?.t_low, 4)}</td></tr>
              <tr><td>PR-AUC (ensemble)</td><td className="mono right">{num(bundle.metrics?.pr_auc_ensemble, 3)}</td></tr>
              <tr><td>PR-AUC (logistic baseline)</td><td className="mono right">{num(bundle.metrics?.pr_auc_logistic_baseline, 3)}</td></tr>
            </tbody></table>
            <div className="tiny muted pad-top">Thresholds come from the institution&rsquo;s cost matrix; Assay claims no universal optimum.</div>
          </div>
        </div>
      )}
      {tab === "bundle" && !bundle && <div className="empty pad">No model bundle recorded.</div>}
    </div>
  );
}

function BundleCard({ bundle }: { bundle: BundleInfo }) {
  return (
    <div className="card" data-testid="bundle-card">
      <div className="card-title">
        Model Bundle &amp; Lineage {bundle.champion && <Pill tone="green">Champion</Pill>}
      </div>
      <div className="kv"><span>Bundle ID</span><span className="mono">{bundle.bundle_id}</span></div>
      <div className="kv"><span>Code commit</span><span className="mono">{bundle.code_commit.slice(0, 12)}</span></div>
      <div className="kv"><span>Dataset manifest</span><span className="mono">{bundle.dataset_id.slice(0, 12)}</span></div>
      <div className="kv"><span>Feature set</span><span className="mono">{bundle.feature_set_version}</span></div>
      <div className="kv"><span>Artefact SHA-256</span><span className="mono">{bundle.artifact_sha256.slice(0, 12)}…</span></div>
      <div className="kv"><span>Trained</span><span className="mono">{bundle.created_at?.slice(0, 19) ?? "n/a"}</span></div>
      <div className="kv"><span>Status</span><span>{bundle.status}</span></div>
    </div>
  );
}
