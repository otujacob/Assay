import { useState } from "react";
import { api } from "../api";
import type { CaseSummaryView } from "../types";
import { Pill } from "./Pill";

const SOURCE_LABEL: Record<string, string> = {
  prediction: "model score", trust_assessment: "trust", explanation: "explanation", policy_decision: "policy", graph: "linked entities",
};

/**
 * A short summary of the case, written by fixed rules from the stored record (the first step of the PRD's copilot). No
 * language model is used and nothing leaves the system. Each line says where it came from. It restates the evidence on this
 * screen; it does not weigh it or advise. Loaded on request.
 */
export function CaseSummary({ decisionId }: { decisionId: string }) {
  const [data, setData] = useState<CaseSummaryView | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setBusy(true);
    setError(null);
    try {
      setData(await api.summary(decisionId));
    } catch (e) {
      setData(null);
      setError(`Could not summarise the case: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card" data-testid="case-summary">
      <div className="card-title">Case summary</div>
      {!data && (
        <>
          <div className="tiny muted">
            The facts on this case in a few lines, each with its source. Written by fixed rules: no AI model is used and nothing leaves the system.
          </div>
          <div className="pad-top">
            <button type="button" className="btn ghost" disabled={busy} onClick={() => void load()}>
              {busy ? "Summarising…" : "Summarise this case"}
            </button>
          </div>
        </>
      )}
      {error && <div className="error" role="alert">{error}</div>}
      {data && (
        <>
          <ul className="cf-list" data-testid="summary-facts">
            {data.facts.map((f, i) => (
              <li key={i} className="small">
                {f.text} <Pill tone="grey" title={f.ref.join(", ") || undefined}>{SOURCE_LABEL[f.source] ?? f.source}</Pill>
              </li>
            ))}
          </ul>
          <div className="tiny muted pad-top" data-testid="summary-not-covered">Not covered: {data.not_covered.join("; ")}.</div>
          <div className="tiny muted pad-top" data-testid="summary-note">{data.note}</div>
        </>
      )}
    </div>
  );
}
