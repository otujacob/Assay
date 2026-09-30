import { useState } from "react";
import { ApiError } from "../api";
import type { ActionRequest, ActionResponse, CaseView } from "../types";
import { ANALYST_REASON_CODES, CHECKLIST, needsReason } from "../lib/format";

type Submit = (body: ActionRequest) => Promise<ActionResponse>;

/**
 * Analyst decision controls (PRD 10.5). Recommend-only: these record a human decision; nothing
 * here acts on a transaction. An override, or disagreeing with the recommendation, needs a reason
 * code. The client mirrors that rule for early feedback, and the SERVER enforces it (FR-26).
 */
export function ActionPanel({ view, onSubmit, canAct }: { view: CaseView; onSubmit: Submit; canAct: boolean }) {
  const [reason, setReason] = useState("");
  const [confidence, setConfidence] = useState(0.7);
  const [notes, setNotes] = useState("");
  const [checks, setChecks] = useState<Record<string, boolean>>({});
  const [overriding, setOverriding] = useState(false);
  const [overrideTo, setOverrideTo] = useState<"approve" | "block">("approve");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<ActionResponse | null>(null);

  const recommendation = view.decision?.recommendation ?? null;
  const closed = view.status === "adjudicated" || (!!view.final_decision && view.status === "decided" && done !== null);

  async function submit(action: ActionRequest["action"]) {
    setError(null);
    if (needsReason(action, recommendation) && !reason) {
      setError(action === "override"
        ? "An override needs a reason code."
        : "Disagreeing with the recommendation needs a reason code.");
      return;
    }
    const body: ActionRequest = {
      action, confidence, checklist: checks,
      ...(reason ? { reason_code: reason } : {}),
      ...(notes ? { notes } : {}),
      ...(action === "override" ? { override_to: overrideTo } : {}),
    };
    setBusy(true);
    try {
      const res = await onSubmit(body);
      setDone(res);
      setOverriding(false);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.message || e.code}` : "Could not record the action.");
    } finally {
      setBusy(false);
    }
  }

  if (!canAct) {
    return <div className="card muted small">Your role can view this case but not record decisions.</div>;
  }
  if (view.status === "adjudicated") {
    return <div className="card small">This case was adjudicated: <b>{view.final_decision}</b>. It is closed.</div>;
  }
  const disabled = busy || closed;
  return (
    <div className="action-panel" data-testid="action-panel">
      <div className="action-buttons">
        <button className="btn approve" disabled={disabled} onClick={() => submit("approve")}>✓ Approve</button>
        <button className="btn block" disabled={disabled} onClick={() => submit("block")}>⊘ Block</button>
        <button className="btn escalate" disabled={disabled} onClick={() => submit("escalate")}>⇪ Escalate</button>
        <button className={`btn override ${overriding ? "on" : ""}`} disabled={disabled}
                onClick={() => setOverriding((o) => !o)} aria-expanded={overriding}>⇄ Override AI</button>
      </div>
      {view.redacted && (
        <div className="tiny muted pad-top">
          Blind review: you decide without the model&rsquo;s score or recommendation. They are shown after you record your decision.
        </div>
      )}
      <div className="card override-card">
        <div className="card-title">{overriding ? "Override Details" : "Decision Details"}</div>
        {overriding && (
          <div className="row gap">
            <span className="small">Override the recommendation to:</span>
            {(["approve", "block"] as const).map((v) => (
              <label key={v} className="radio">
                <input type="radio" name="override-to" checked={overrideTo === v} onChange={() => setOverrideTo(v)} /> {v}
              </label>
            ))}
          </div>
        )}
        <div className="form-grid">
          <label>
            Reason Code{(overriding || needsReason("approve", recommendation) || needsReason("block", recommendation)) ? " *" : ""}
            <select aria-label="Reason code" value={reason} onChange={(e) => setReason(e.target.value)}>
              <option value="">Select reason code</option>
              {ANALYST_REASON_CODES.map((r) => <option key={r.code} value={r.code}>{r.label}</option>)}
            </select>
          </label>
          <label>
            Confidence: {Math.round(confidence * 100)}%
            <input aria-label="Confidence" type="range" min={0} max={1} step={0.05} value={confidence}
                   onChange={(e) => setConfidence(Number(e.target.value))} />
          </label>
        </div>
        <div className="checklist">
          {CHECKLIST.map((c) => (
            <label key={c.key} className="check">
              <input type="checkbox" checked={!!checks[c.key]} onChange={(e) => setChecks({ ...checks, [c.key]: e.target.checked })} /> {c.label}
            </label>
          ))}
        </div>
        <label className="block-label">
          Evidence Checklist &amp; Notes
          <textarea aria-label="Notes" rows={2} placeholder="Add notes, link evidence, or justification…" value={notes} onChange={(e) => setNotes(e.target.value)} />
        </label>
        <div className="tiny muted">Do not put personal data in notes; they are stored in the audit record.</div>
        {overriding && (
          <button className="btn override-submit" disabled={disabled} onClick={() => submit("override")}>Submit override</button>
        )}
        <div className="row gap pad-top">
          <button className="btn ghost" disabled={disabled} onClick={() => submit("unsure")}>Mark unsure</button>
          <button className="btn ghost" disabled={disabled} onClick={() => submit("request_review")}>Request second review</button>
        </div>
        {error && <div className="error" role="alert">{error}</div>}
      </div>
      {done && (
        <div className="card feedback" data-testid="action-result" role="status">
          <div className="card-title">Recorded</div>
          <div className="small">Case status: <b>{done.status}</b>{done.final_decision ? ` · decision: ${done.final_decision}` : ""}
            {done.needs_second_review ? " · goes to second review (override of a high-trust recommendation)" : ""}</div>
          <div className="tiny muted pad-top">
            Feedback quality (shadow, not used for learning): FQS {done.feedback.fqs.toFixed(2)} ·{" "}
            {done.feedback.disposition} ({done.feedback.reason}) · formula {done.feedback.formula_version}
          </div>
        </div>
      )}
    </div>
  );
}
