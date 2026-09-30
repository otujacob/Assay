import type { Dashboard } from "../types";
import { formatSla, pct } from "../lib/format";
import { Pill } from "./Pill";

function Card({ icon, label, children, tone }: { icon: string; label: string; children: React.ReactNode; tone: string }) {
  return (
    <div className="kpi">
      <div className={`kpi-icon ${tone}`} aria-hidden="true">{icon}</div>
      <div className="kpi-body">
        <div className="kpi-label">{label}</div>
        {children}
      </div>
    </div>
  );
}

/**
 * The four headline figures. Numbers that cannot be computed yet are shown as unavailable, with
 * why, instead of a placeholder value: a live system has no matured outcomes on day one, so there
 * is no high-trust error rate to report yet (PRD 6, 21).
 */
export function KpiCards({ dash, elapsedSeconds }: { dash: Dashboard | null; elapsedSeconds: number }) {
  if (!dash) {
    return <div className="kpis"><div className="kpi muted small">Dashboard figures are available to the manager role.</div></div>;
  }
  const p = dash.pending_reviews;
  const ht = dash.high_trust_error_rate;
  const ie = dash.insufficient_evidence_rate;
  const fa = dash.feedback_acceptance;
  const delta = ie.value !== null && ie.previous !== null ? ie.value - ie.previous : null;
  return (
    <div className="kpis" data-testid="kpis">
      <Card icon="▤" label="Total Pending Reviews" tone="blue">
        <div className="kpi-value">{p.count}
          {p.sla_breached > 0 && <Pill tone="red" title="Cases past their SLA">{p.sla_breached} past SLA</Pill>}
        </div>
        <div className="tiny muted">
          of {p.total_cases} cases ever queued
          {p.soonest_sla_seconds === null ? "" : p.soonest_sla_seconds - elapsedSeconds < 0
            ? ` · most overdue ${formatSla(-(p.soonest_sla_seconds - elapsedSeconds))}`
            : ` · next SLA ${formatSla(p.soonest_sla_seconds - elapsedSeconds)}`}
        </div>
      </Card>
      <Card icon="⛨" label="High-Trust Error Rate" tone="green">
        {ht.value === null ? (
          <>
            <div className="kpi-value small-value">Not yet available</div>
            <div className="tiny muted">{ht.basis}</div>
          </>
        ) : (
          <>
            <div className="kpi-value">{pct(ht.value, 2)} <Pill tone="green">Matured Outcomes</Pill></div>
            <div className="tiny muted">95% CI {pct(ht.lo, 2)} to {pct(ht.hi, 2)} · n={ht.n}</div>
          </>
        )}
      </Card>
      <Card icon="⚠" label="Insufficient Evidence Rate" tone="purple">
        <div className="kpi-value">{pct(ie.value)} <Pill tone="grey">Auto-Routed</Pill></div>
        <div className="tiny muted">
          {delta === null ? `no earlier window to compare (n=${ie.n})`
            : `${delta >= 0 ? "▲" : "▼"} ${pct(Math.abs(delta))} vs previous ${ie.window_days} days`}
        </div>
      </Card>
      <Card icon="✉" label="Feedback Acceptance Rate" tone="blue">
        <div className="kpi-value">{fa.acceptance_rate === null ? "n/a" : pct(fa.acceptance_rate)}</div>
        <div className="tiny muted">
          {fa.accepted} accepted · {fa.rejected} rejected · {fa.deferred} deferred (shadow; nothing learns from it yet)
        </div>
      </Card>
    </div>
  );
}
