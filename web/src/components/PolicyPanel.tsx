import { useCallback, useEffect, useState } from "react";
import { api, type PolicyRequest } from "../api";
import type { PolicyPayload, PolicyPreview, PolicyVersion } from "../types";
import { formatTime } from "../lib/format";
import { Pill } from "./Pill";
import { PolicyPreviewCard } from "./PolicyPreviewCard";

const DQ_LABEL: Record<string, string> = {
  request_human_review: "Send to human review",
  hold: "Hold the transaction",
};

export interface PolicyForm { dq: string; from: string; tLow: string; tHigh: string; amount: string }
export const EMPTY_FORM: PolicyForm = { dq: "request_human_review", from: "", tLow: "", tHigh: "", amount: "" };

/** The form as a request. Blank fields are left out, so the model's own values apply. Returns an error
 *  message instead when what was typed cannot be a policy; the server checks again. */
export function buildRequest(f: PolicyForm): PolicyRequest | string {
  const req: PolicyRequest = { dq_gate_action: f.dq };
  const num = (s: string): number => (s.trim() === "" ? NaN : Number(s));
  if ((f.tLow.trim() === "") !== (f.tHigh.trim() === "")) return "Set both risk thresholds, or neither.";
  if (f.tLow.trim() !== "") {
    const lo = num(f.tLow), hi = num(f.tHigh);
    if (!(Number.isFinite(lo) && Number.isFinite(hi) && lo > 0 && lo < hi && hi < 1)) {
      return "Risk thresholds must be numbers with 0 < low < high < 1.";
    }
    req.t_low = lo;
    req.t_high = hi;
  }
  if (f.amount.trim() !== "") {
    const a = num(f.amount);
    if (!(Number.isFinite(a) && a > 0)) return "The amount must be a number above zero.";
    req.always_review_above = a;
  }
  if (f.from) req.effective_from = `${f.from}T00:00:00+00:00`;
  return req;
}

const money = (n: number) => n.toLocaleString("en-GB");

/** A policy's rules in one line. */
export function summarise(p: PolicyPayload): string {
  const parts = [p.dq_gate_action === "hold" ? "Hold on poor data" : "Review on poor data"];
  if (p.t_low !== undefined && p.t_high !== undefined) parts.unshift(`Risk bands ${p.t_low} / ${p.t_high}`);
  if (p.always_review_above !== undefined) parts.push(`Review amounts from ${money(p.always_review_above)}`);
  return parts.join(" · ");
}

/** When an approved policy takes effect: the later of its effective date and its approval time (the server's rule). */
const takesEffect = (p: PolicyVersion): number =>
  Math.max(Date.parse(p.effective_from), p.approved_at ? Date.parse(p.approved_at) : Infinity);

/** The approved policy in force at `now`; of several, the one that took effect last (the newest wins a tie). */
export function inForce(items: PolicyVersion[], now: number = Date.now()): PolicyVersion | undefined {
  let best: PolicyVersion | undefined;
  for (const p of [...items].reverse()) { // the list is newest first, so walk oldest to newest
    if (p.status === "approved" && takesEffect(p) <= now && (!best || takesEffect(p) >= takesEffect(best))) best = p;
  }
  return best;
}

/** Decision policies (PRD 10.4, FR-21): one person proposes, a different person approves. */
export function PolicyPanel({ userKey, roles }: { userKey: string; roles: ReadonlySet<string> }) {
  const [items, setItems] = useState<PolicyVersion[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<PolicyForm>(EMPTY_FORM);
  const [preview, setPreview] = useState<PolicyPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const set = (k: keyof PolicyForm) => (e: { target: { value: string } }) => {
    setForm({ ...form, [k]: e.target.value });
    setPreview(null); // a preview describes the form as it was, so editing the form clears it
  };
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

  const active = items ? inForce(items) : undefined;

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
        <form className="filters filters-row" onSubmit={(e) => {
          e.preventDefault();
          const req = buildRequest(form);
          if (typeof req === "string") { setError(req); return; }
          void run(async () => { await api.proposePolicy(req); setPreview(null); setForm(EMPTY_FORM); });
        }}>
          <label className="field-wide">When data quality is below the floor
            <select aria-label="Data-quality gate action" value={form.dq} onChange={set("dq")}>
              {Object.entries(DQ_LABEL).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </label>
          <label>Low risk below
            <input aria-label="Low risk below" inputMode="decimal" placeholder="model default" value={form.tLow} onChange={set("tLow")} />
          </label>
          <label>High risk from
            <input aria-label="High risk from" inputMode="decimal" placeholder="model default" value={form.tHigh} onChange={set("tHigh")} />
          </label>
          <label>Always review amounts from
            <input aria-label="Always review amounts from" inputMode="decimal" placeholder="no limit" value={form.amount} onChange={set("amount")} />
          </label>
          <label>Effective from
            <input aria-label="Effective from" type="date" value={form.from} onChange={set("from")} />
          </label>
          {(form.tLow.trim() !== "" || form.tHigh.trim() !== "") && (
            <div className="tiny muted threshold-note" role="note" style={{ flexBasis: "100%" }}>
              Risk thresholds set here replace the model&rsquo;s own. The validation figures under Governance were measured
              at the model&rsquo;s thresholds, so they do not describe a policy that moves them.
            </div>
          )}
          <div className="actions">
            <button type="button" className="btn ghost" disabled={busy} onClick={() => {
              const req = buildRequest(form);
              if (typeof req === "string") { setError(req); return; }
              setBusy(true);
              api.previewPolicy(req).then((p) => { setPreview(p); setError(null); })
                .catch((e: Error) => setError(e.message)).finally(() => setBusy(false));
            }}>Preview impact</button>
            <button type="submit" className="btn primary" disabled={busy}>Propose policy</button>
          </div>
        </form>
      )}
      {preview && <div className="pad"><PolicyPreviewCard preview={preview} /></div>}

      <div className="table-wrap">
        <table className="queue">
          <thead>
            <tr><th>Version</th><th>Rules</th><th>Effective from (UTC)</th><th>Proposed by</th><th>Status</th><th /></tr>
          </thead>
          <tbody>
            {(items ?? []).map((p) => {
              const own = p.proposed_by === me;
              return (
                <tr key={p.id}>
                  <td className="mono">{p.version}{p.id === active?.id && <> <Pill tone="green">In force</Pill></>}</td>
                  <td className="small">{summarise(p.payload)}</td>
                  <td className="mono">{formatTime(p.effective_from)}</td>
                  <td className="mono">{p.proposed_by}</td>
                  <td>
                    {p.status === "pending"
                      ? <Pill tone="amber">Awaiting approval</Pill>
                      : <>
                          <Pill tone="green" title={p.approved_at ?? undefined}>Approved by {p.approved_by}</Pill>
                          {takesEffect(p) > Date.now() && <> <Pill tone="blue">Scheduled</Pill></>}
                        </>}
                  </td>
                  <td className="right">
                    {p.status === "pending" && canApprove && (
                      <button type="button" className="btn ghost" disabled={busy || own}
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
