import type { Explanation } from "../types";
import { GROUP_LABEL, num } from "../lib/format";
import { Pill } from "./Pill";

/** Display thresholds only: placeholders, not validated (PRD 7). */
export const stabilityLabel = (s: number | null): { text: string; tone: "green" | "amber" | "red" | "grey" } =>
  s === null ? { text: "Unavailable", tone: "grey" } : s >= 0.8 ? { text: "High stability", tone: "green" }
    : s >= 0.6 ? { text: "Moderate stability", tone: "amber" } : { text: "Low stability", tone: "red" };

export function ShapDrivers({ explanation, pending }: { explanation: Explanation | null; pending: boolean }) {
  if (!explanation) {
    return (
      <div className="card" data-testid="shap-pending">
        <div className="card-title">Explainability Engine (TreeSHAP)</div>
        <div className="muted small">
          {pending
            ? "Explanation testing has not run for this case yet. It runs asynchronously, and the Trust Index is capped below High until it does."
            : "No explanation available."}
        </div>
      </div>
    );
  }
  const rows = Object.entries(explanation.attributions)
    .map(([k, v]) => ({ key: k, label: GROUP_LABEL[k] ?? k, v }))
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const max = Math.max(0.0001, ...rows.map((r) => Math.abs(r.v)));
  const st = stabilityLabel(explanation.stability);
  return (
    <div className="card" data-testid="shap-drivers">
      <div className="card-title">Explainability Engine (TreeSHAP)</div>
      <div className="shap-grid">
        <div>
          <div className="tiny muted pad-bottom">Top feature drivers, in log-odds. Red pushes toward fraud, teal toward legitimate.</div>
          {rows.map((r) => (
            <div className="shap-row" key={r.key} data-testid={`shap-${r.key}`}>
              <div className="shap-label">{r.label}</div>
              <div className="shap-track">
                <div className={`shap-bar ${r.v >= 0 ? "pos" : "neg"}`} style={{ width: `${(Math.abs(r.v) / max) * 100}%` }} />
              </div>
              <div className="shap-val">{r.v >= 0 ? "+" : ""}{num(r.v)}</div>
            </div>
          ))}
        </div>
        <div className="stability-box">
          <div className="stab-title">
            Attribution stability: {num(explanation.stability)}
          </div>
          <Pill tone={st.tone}>{st.text}</Pill>
          <div className="tiny muted pad-top">
            Sensitivity {num(explanation.sensitivity)} · Faithfulness {num(explanation.faithfulness)} ·{" "}
            {explanation.reproducible ? "Reproducible" : "NOT reproducible"}
          </div>
          <div className="tiny muted pad-top">
            Explains the boosted-tree members only. A stable, faithful explanation can still explain a wrong prediction.
          </div>
        </div>
      </div>
    </div>
  );
}
