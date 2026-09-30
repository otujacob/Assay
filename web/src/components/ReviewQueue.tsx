import { useMemo } from "react";
import type { QueueItem } from "../types";
import { formatSla, formatTime, money, slaTone } from "../lib/format";
import { PriorityPill, RiskPill, TrustPill } from "./Pill";

export interface QueueFilterState {
  risk: string;
  trust: string;
  reason: string;
  search: string;
}

export const EMPTY_FILTERS: QueueFilterState = { risk: "", trust: "", reason: "", search: "" };

/** Filtering happens on the server (it knows what is redacted); this only narrows what is shown. */
export function ReviewQueue({
  items, selectedId, onSelect, filters, onFilters, elapsedSeconds, total, loading,
}: {
  items: QueueItem[];
  selectedId: string | null;
  onSelect: (decisionId: string) => void;
  filters: QueueFilterState;
  onFilters: (f: QueueFilterState) => void;
  elapsedSeconds: number;
  total: number;
  loading: boolean;
}) {
  const reasons = useMemo(() => Array.from(new Set(items.flatMap((i) => i.reason_codes))).sort(), [items]);
  const set = (k: keyof QueueFilterState) => (e: { target: { value: string } }) => onFilters({ ...filters, [k]: e.target.value });
  return (
    <div className="panel" data-testid="review-queue">
      <div className="panel-head">
        <div className="panel-title">Review Queue</div>
        <div className="filters">
          <label>Risk Band
            <select aria-label="Filter by risk band" value={filters.risk} onChange={set("risk")}>
              <option value="">All</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option>
            </select>
          </label>
          <label>Trust Band
            <select aria-label="Filter by trust band" value={filters.trust} onChange={set("trust")}>
              <option value="">All</option><option value="high">High</option><option value="moderate">Moderate</option>
              <option value="low">Low</option><option value="insufficient_evidence">Insufficient Evidence</option>
            </select>
          </label>
          <label>Reason Codes
            <select aria-label="Filter by reason code" value={filters.reason} onChange={set("reason")}>
              <option value="">All</option>
              {reasons.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          </label>
          <input aria-label="Search transactions" className="search" placeholder="Search transactions…" value={filters.search} onChange={set("search")} />
        </div>
      </div>
      <div className="table-wrap">
        <table className="queue">
          <thead>
            <tr>
              <th>Priority</th><th>Txn ID</th><th>Risk Band</th><th>Trust Band</th>
              <th className="right">Amount</th><th>Time (UTC)</th><th>Queue SLA</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it) => {
              const left = it.sla.seconds_left - elapsedSeconds;
              const tone = slaTone(left, it.sla.minutes);
              return (
                <tr key={it.decision_id} className={it.decision_id === selectedId ? "selected" : ""}
                    onClick={() => onSelect(it.decision_id)} data-testid={`row-${it.txn_id}`}
                    tabIndex={0} onKeyDown={(e) => { if (e.key === "Enter") onSelect(it.decision_id); }}>
                  <td><PriorityPill label={it.priority} /></td>
                  <td className="mono">{it.txn_id}{it.blind && <span className="blind-dot" title="Blind review: score and recommendation are hidden from analysts"> ◐</span>}</td>
                  <td>{it.redacted || !it.risk_band ? <span className="muted">hidden</span> : <RiskPill band={it.risk_band} />}</td>
                  <td>{it.redacted || !it.trust_state ? <span className="muted">hidden</span> : <TrustPill state={it.trust_state} />}</td>
                  <td className="right mono">{money(it.amount, it.currency)}</td>
                  <td className="mono">{formatTime(it.event_time)}</td>
                  <td><span className={`sla sla-${tone}`} title={it.sla.breached ? "SLA breached" : "Time left"}>{formatSla(left)}</span></td>
                </tr>
              );
            })}
            {!loading && items.length === 0 && (
              <tr><td colSpan={7} className="empty">No cases match. The queue only holds decisions that need a human.</td></tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="panel-foot tiny muted">
        Showing {items.length} of {total} cases · ordered by priority (risk, amount, trust and age)
      </div>
    </div>
  );
}
