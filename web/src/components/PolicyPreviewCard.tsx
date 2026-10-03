import type { PolicyPreview } from "../types";
import { formatTime } from "../lib/format";

const ACTION_LABEL: Record<string, string> = {
  approve: "Approve",
  approve_sampled_qa: "Approve (sampled for QA)",
  block: "Block",
  escalate: "Escalate",
  request_human_review: "Human review",
  request_human_review_priority: "Priority human review",
  hold: "Hold",
};
const label = (a: string) => ACTION_LABEL[a] ?? a;
const REASON: Record<string, string> = {
  segment_rule: "amount rule", matrix: "risk/trust matrix", data_quality: "data-quality gate", novelty: "novelty gate",
};

/** The size of the review-volume change, in words a manager can act on. */
export function volumeSummary(v: PolicyPreview["review_volume"]): string {
  const sign = v.change > 0 ? "+" : "";
  const pct = v.change_pct === null ? "" : `, ${sign}${v.change_pct}%`;
  if (v.change === 0) return `No change in review volume (${v.before} cases${v.change_pct === null ? "" : ", 0%"})`;
  return `Review volume ${v.before} → ${v.after} (${sign}${v.change}${pct})`;
}

/** What a proposed policy would have done to past cases, compared with the policy in force (PRD 10.4). */
export function PolicyPreviewCard({ preview }: { preview: PolicyPreview }) {
  const v = preview.review_volume;
  const tone = v.change > 0 ? "up" : v.change < 0 ? "down" : "same";
  return (
    <div className="card preview-card" data-testid="policy-preview">
      <div className="card-title">Impact on past cases</div>
      <div className={`preview-headline preview-${tone}`}>{volumeSummary(v)}</div>
      <div className="tiny muted">
        {preview.n_decisions} past cases{preview.window.from ? `, ${formatTime(preview.window.from)} to ${formatTime(preview.window.to ?? preview.window.from)}` : ""},
        compared with {preview.baseline_version}, the policy in force.
        {preview.skipped > 0 && ` ${preview.skipped} could not be replayed.`}
      </div>
      {preview.n_decisions === 0 ? (
        <div className="small pad-top">There are no past decisions to replay yet.</div>
      ) : preview.changed_decisions === 0 ? (
        <div className="small pad-top">No decision would have changed.</div>
      ) : (
        <>
          <table className="mini pad-top">
            <thead><tr><th>Was</th><th>Would be</th><th className="right">Cases</th></tr></thead>
            <tbody>
              {preview.changes.map((c) => (
                <tr key={`${c.from}>${c.to}`}><td>{label(c.from)}</td><td>{label(c.to)}</td><td className="right mono">{c.count}</td></tr>
              ))}
            </tbody>
          </table>
          {preview.examples.length > 0 && (
            <details className="small pad-top">
              <summary>Examples ({preview.examples.length})</summary>
              <table className="mini">
                <thead><tr><th>Transaction</th><th className="right">Amount</th><th className="right">Risk</th><th>Was</th><th>Would be</th><th>Because</th></tr></thead>
                <tbody>
                  {preview.examples.map((e) => (
                    <tr key={e.txn_id}>
                      <td className="mono">{e.txn_id}</td><td className="right mono">{e.amount.toLocaleString("en-GB")}</td>
                      <td className="right mono">{e.risk.toFixed(2)}</td><td>{label(e.was)}</td><td>{label(e.now)}</td>
                      <td className="tiny">{REASON[e.because] ?? e.because}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          )}
        </>
      )}
      <ul className="limits tiny muted pad-top">
        {preview.caveats.map((c) => <li key={c}>{c}</li>)}
      </ul>
    </div>
  );
}
