import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { PolicyVersion } from "../types";
import { formatTime } from "../lib/format";
import { Pill } from "./Pill";

const DQ_LABEL: Record<string, string> = {
  request_human_review: "Send to human review",
  hold: "Hold the transaction",
};

/** Decision policies (PRD 10.4, FR-21): one person proposes, a different person approves. */
export function PolicyPanel({ userKey, roles }: { userKey: string; roles: ReadonlySet<string> }) {
  const [items, setItems] = useState<PolicyVersion[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dq, setDq] = useState("request_human_review");
  const [from, setFrom] = useState("");
  const [busy, setBusy] = useState(false);
  const me = `u:${userKey}`;
  const canPropose = roles.has("admin");
  const canApprove = roles.has("approver");

  const load = useCallback(async () => {
    try {
      setItems((await api.policies()).items);
      setError(null);
    } catch (e) {
      setItems(null);
      setError(`Could not load policies: ${(e as Error).message}`);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const active = items?.find((p) => p.status === "approved");

  return (
    <div className="panel" data-testid="policy-panel">
      <div className="panel-head">
        <div className="panel-title">Decision Policies</div>
      </div>
      <div className="gov-note tiny">
        A policy takes effect only once a <b>different</b> person approves it, and never before its effective date.
        Until one is approved, the built-in default (policy-0) applies. Automation stays at level 0: Assay recommends, people decide.
      </div>
      {error && <div className="banner-error" role="alert">{error}</div>}

      {canPropose && (
        <form className="filters pad" onSubmit={(e) => {
          e.preventDefault();
          void run(() => api.proposePolicy({
            dq_gate_action: dq, effective_from: from ? `${from}T00:00:00+00:00` : undefined,
          }));
        }}>
          <label className="small">When data quality is below the floor{" "}
            <select aria-label="Data-quality gate action" value={dq} onChange={(e) => setDq(e.target.value)}>
              {Object.entries(DQ_LABEL).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </label>
          <label className="small">Effective from{" "}
            <input aria-label="Effective from" type="date" value={from} onChange={(e) => setFrom(e.target.value)} />
          </label>
          <button type="submit" disabled={busy}>Propose policy</button>
        </form>
      )}

      <div className="table-wrap">
        <table className="queue">
          <thead>
            <tr><th>Version</th><th>Data-quality gate</th><th>Effective from (UTC)</th><th>Proposed by</th><th>Status</th><th /></tr>
          </thead>
          <tbody>
            {(items ?? []).map((p) => {
              const own = p.proposed_by === me;
              return (
                <tr key={p.id}>
                  <td className="mono">{p.version}{p.id === active?.id && <> <Pill tone="green">In force</Pill></>}</td>
                  <td>{DQ_LABEL[p.payload.dq_gate_action] ?? p.payload.dq_gate_action}</td>
                  <td className="mono">{formatTime(p.effective_from)}</td>
                  <td className="mono">{p.proposed_by}</td>
                  <td>
                    {p.status === "approved"
                      ? <Pill tone="green" title={p.approved_at ?? undefined}>Approved by {p.approved_by}</Pill>
                      : <Pill tone="amber">Awaiting approval</Pill>}
                  </td>
                  <td className="right">
                    {p.status === "pending" && canApprove && (
                      <button type="button" disabled={busy || own}
                              title={own ? "You proposed this policy, so someone else must approve it" : undefined}
                              onClick={() => void run(() => api.approvePolicy(p.id))}>
                        Approve
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
            {items && items.length === 0 && (
              <tr><td colSpan={6} className="empty">No policies yet. The built-in default (policy-0) is in force.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
