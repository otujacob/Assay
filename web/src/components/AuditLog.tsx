import { useCallback, useEffect, useState } from "react";
import { api, type AuditFilters } from "../api";
import type { AuditResponse } from "../types";
import { formatTime } from "../lib/format";
import { Pill } from "./Pill";

const EMPTY = { txn: "", actor: "", action: "", from: "", to: "" };
type Form = typeof EMPTY;

const addDay = (d: string): string => {
  const t = new Date(`${d}T00:00:00Z`);
  t.setUTCDate(t.getUTCDate() + 1);
  return t.toISOString().slice(0, 10);
};

/** Form fields to API filters. The "to" date is inclusive, so the API's exclusive bound is the next day. */
export const toFilters = (f: Form): AuditFilters => ({
  txn_id: f.txn.trim() || undefined,
  actor: f.actor.trim() || undefined,
  action: f.action.trim() || undefined,
  since: f.from ? `${f.from}T00:00:00` : undefined,
  until: f.to ? `${addDay(f.to)}T00:00:00` : undefined,
});

/** Read-only audit view (FR-34): search by transaction, actor, action and date, and export. */
export function AuditLog() {
  const [form, setForm] = useState<Form>(EMPTY);
  const [data, setData] = useState<AuditResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (f: Form) => {
    try {
      setData(await api.audit({ ...toFilters(f), limit: 200 }));
      setError(null);
    } catch (e) {
      setData(null);
      setError(`Could not load the audit log: ${(e as Error).message}`);
    }
  }, []);

  useEffect(() => {
    void load(EMPTY);
  }, [load]);

  const set = (k: keyof Form) => (e: { target: { value: string } }) => setForm({ ...form, [k]: e.target.value });

  const exportCsv = async () => {
    try {
      const url = URL.createObjectURL(await api.auditCsv(toFilters(form)));
      const a = document.createElement("a");
      a.href = url;
      a.download = "assay-audit.csv";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(`Could not export the audit log: ${(e as Error).message}`);
    }
  };

  return (
    <div className="panel" data-testid="audit-log">
      <div className="panel-head">
        <div className="panel-title">Audit Log</div>
        <form className="filters" onSubmit={(e) => { e.preventDefault(); void load(form); }}>
          <input aria-label="Transaction" className="search" placeholder="Transaction id" value={form.txn} onChange={set("txn")} />
          <input aria-label="Actor" className="search" placeholder="Actor" value={form.actor} onChange={set("actor")} />
          <input aria-label="Action" className="search" placeholder="Action" value={form.action} onChange={set("action")} />
          <input aria-label="From date" type="date" value={form.from} onChange={set("from")} />
          <input aria-label="To date" type="date" value={form.to} onChange={set("to")} />
          <button type="submit">Search</button>
          <button type="button" onClick={() => void exportCsv()}>Export CSV</button>
        </form>
      </div>
      {error && <div className="banner-error" role="alert">{error}</div>}
      <div className="table-wrap">
        <table className="queue">
          <thead>
            <tr><th>Time (UTC)</th><th>Actor</th><th>Action</th><th>Object</th><th>Result</th></tr>
          </thead>
          <tbody>
            {(data?.items ?? []).map((r) => (
              <tr key={r.row_hash}>
                <td className="mono">{formatTime(r.time)}</td>
                <td className="mono">{r.actor}</td>
                <td>{r.action}</td>
                <td className="mono">{r.object}</td>
                <td>{r.result}</td>
              </tr>
            ))}
            {data && data.items.length === 0 && <tr><td colSpan={5} className="empty">No records match.</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="panel-foot tiny muted">
        {data && (
          <>
            Showing {data.items.length} of {data.matching}.{" "}
            <Pill tone={data.chain_ok ? "green" : "red"}>
              {data.chain_ok ? "Hash chain verified" : "Hash chain BROKEN"}
            </Pill>{" "}
            Reading this log is itself recorded.
          </>
        )}
      </div>
    </div>
  );
}
