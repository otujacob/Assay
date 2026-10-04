import { useState } from "react";
import type { ActionRequest, ActionResponse, CaseView } from "../types";
import { RECOMMENDATION_LABEL, formatTime, money, num, pct } from "../lib/format";
import { ActionPanel } from "./ActionPanel";
import { ComponentBars } from "./ComponentBars";
import { CounterfactualCard } from "./CounterfactualCard";
import { Pill, PriorityPill, RiskPill } from "./Pill";
import { ShapDrivers } from "./ShapDrivers";
import { TrustGauge } from "./TrustGauge";

export function CaseDetails({
  view, canAct, canAdjudicate, onAct, onAdjudicate, onBack,
}: {
  view: CaseView;
  canAct: boolean;
  canAdjudicate: boolean;
  onAct: (body: ActionRequest) => Promise<ActionResponse>;
  onAdjudicate: (final: "approve" | "block", rationale: string) => Promise<void>;
  onBack: () => void;
}) {
  const d = view.decision;
  const t = view.transaction;
  return (
    <div className="case" data-testid="case-details">
      <div className="case-head">
        <div>
          <div className="case-id">
            <span className="mono">{view.txn_id}</span>
            <PriorityPill label={view.priority.label} />
            {view.blind && <Pill tone="purple" title="Decided once when the case entered the queue">Blind review</Pill>}
          </div>
          <div className="muted small">
            {money(t.amount, t.currency)} · {t.channel.replace(/_/g, " ")} · {formatTime(t.event_time)} UTC
            {t.country ? ` · ${t.country}` : ""}
          </div>
          <div className="muted tiny">
            Customer {t.customer ?? "n/a"}{t.beneficiary ? ` · Beneficiary ${t.beneficiary}` : ""} (masked)
          </div>
        </div>
        <button className="btn ghost" onClick={onBack}>‹ Back to Queue</button>
      </div>

      {view.outcome && (
        <div className="notice info">
          Verified outcome: <b>{view.outcome.verified}</b> ({view.outcome.type}). A verified outcome, not an analyst decision, is the label.
        </div>
      )}

      {view.redacted || !d ? (
        <div className="notice blind" data-testid="blind-notice">
          <b>Blind review.</b> The model&rsquo;s score, Trust Index and recommendation are hidden until you record your
          decision, so your judgement is independent. Transaction details above are available.
        </div>
      ) : (
        <>
          <div className="two-col">
            <div className="card risk-card" data-testid="risk-card">
              <div className="card-title">Fraud Risk Score</div>
              <div className="risk-number">{num(d.risk)}</div>
              <div className="row gap"><RiskPill band={d.risk_band} /></div>
              <div className="risk-bar"><div style={{ width: `${Math.min(100, d.risk * 100)}%` }} /></div>
              <div className="tiny muted pad-top">
                Calibrated probability ({pct(d.risk, 1)}). Not the Trust Index: it says how likely fraud is, not how far to rely on it.
              </div>
            </div>
            <TrustGauge trust={d.trust} />
          </div>
          <ComponentBars trust={d.trust} />
          <ShapDrivers explanation={view.explanation} pending={d.explanation_status === "pending"} />
          <CounterfactualCard key={view.decision_id} decisionId={view.decision_id} />
          <div className="card recommended" data-testid="recommendation">
            <div className="rec-title">Recommended Action</div>
            <div className="rec-action">{RECOMMENDATION_LABEL[d.recommendation] ?? d.recommendation}</div>
            <div className="tiny muted">
              Decision policy matrix result ({d.gate}) · policy {d.policy_version} · automation level {d.automation_level}: recommend-only, nothing is done automatically.
            </div>
          </div>
        </>
      )}

      {view.needs_second_review && (
        <div className="notice warn">An override of a high-trust recommendation goes to second review (PRD 11.3).</div>
      )}
      {view.status === "conflicted" && (
        <div className="notice warn" data-testid="conflict-notice">
          <b>Conflicting decisions.</b> Analysts disagreed. The case is excluded from learning until a senior analyst adjudicates it.
          {canAdjudicate && <AdjudicationForm onSubmit={onAdjudicate} />}
        </div>
      )}
      {view.status === "escalated" && canAdjudicate && (
        <div className="notice info" data-testid="escalated-notice">
          Escalated to a senior analyst.
          <AdjudicationForm onSubmit={onAdjudicate} />
        </div>
      )}

      <ActionPanel key={view.decision_id} view={view} canAct={canAct} onSubmit={onAct} />

      {view.actions.length > 0 && (
        <div className="card" data-testid="action-history">
          <div className="card-title">Decisions on this case</div>
          {view.actions.map((a, i) => (
            <div className="hist-row small" key={i}>
              <span className="mono">{a.analyst}</span> <span className="muted">({a.role})</span>: <b>{a.action}</b>
              {a.final_decision ? ` → ${a.final_decision}` : ""}
              {a.reason_code ? <span className="muted"> · {a.reason_code}</span> : null}
              <span className="muted tiny"> · {formatTime(a.at)}</span>
            </div>
          ))}
          <div className="tiny muted pad-top">Every decision is kept; none is overwritten.</div>
        </div>
      )}
    </div>
  );
}

function AdjudicationForm({ onSubmit }: { onSubmit: (final: "approve" | "block", rationale: string) => Promise<void> }) {
  const [rationale, setRationale] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const go = async (final: "approve" | "block") => {
    if (!rationale.trim()) {
      setErr("An adjudication needs a written rationale.");
      return;
    }
    setErr(null);
    try {
      await onSubmit(final, rationale);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Could not adjudicate.");
    }
  };
  return (
    <div className="adjudicate">
      <textarea aria-label="Adjudication rationale" rows={2} placeholder="Rationale (stored with the decision)" value={rationale} onChange={(e) => setRationale(e.target.value)} />
      <div className="row gap">
        <button className="btn approve" onClick={() => go("approve")}>Adjudicate: approve</button>
        <button className="btn block" onClick={() => go("block")}>Adjudicate: block</button>
      </div>
      {err && <div className="error" role="alert">{err}</div>}
    </div>
  );
}
