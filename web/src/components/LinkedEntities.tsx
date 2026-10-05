import { useState } from "react";
import { api } from "../api";
import type { GraphLink, GraphView } from "../types";
import { formatTime } from "../lib/format";
import { Pill } from "./Pill";

/** A derived link's underlying relationship, which is what an analyst flags. */
const BASE_REL: Record<string, string> = {
  SHARES_DEVICE: "USED_DEVICE", SHARES_IP: "ACCESSED_FROM", SHARES_BENEFICIARY: "PAID", SHARES_MERCHANT: "TRANSACTED_AT",
};
const pct0 = (x: number) => `${Math.round(x * 100)}%`;

/**
 * What else is linked to this case through shared devices, IP addresses and beneficiaries, as the graph stood when the
 * decision was made (PRD 13). Loaded on request. Investigator context only: a link is evidence of a connection, not of
 * wrongdoing, and the server's caveat is always shown with it. Never for a customer.
 */
export function LinkedEntities({ decisionId, canFlag }: { decisionId: string; canFlag: boolean }) {
  const [data, setData] = useState<GraphView | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setBusy(true);
    setError(null);
    try {
      setData(await api.graph(decisionId));
    } catch (e) {
      setData(null);
      setError(`Could not load linked entities: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card" data-testid="linked-entities">
      <div className="card-title">Linked entities</div>
      {!data && (
        <>
          <div className="tiny muted">
            Other customers who used the same device, IP address or beneficiary, and how sure the graph is of each link. For
            investigators: it shows connections, not guilt.
          </div>
          <div className="pad-top">
            <button type="button" className="btn ghost" disabled={busy} onClick={() => void load()}>
              {busy ? "Loading…" : "Show linked entities"}
            </button>
          </div>
        </>
      )}
      {error && <div className="error" role="alert">{error}</div>}
      {data && <View data={data} canFlag={canFlag} />}
    </div>
  );
}

export function View({ data, canFlag }: { data: GraphView; canFlag: boolean }) {
  const [flagged, setFlagged] = useState<Set<string>>(new Set());
  const [open, setOpen] = useState<string | null>(null);
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const key = (l: GraphLink) => `${l.other_customer}|${l.via}`;

  const submit = async (l: GraphLink) => {
    setBusy(true);
    setError(null);
    try {
      await api.flagEdge({ relationship: BASE_REL[l.relationship] ?? l.relationship, src: l.other_customer, dst: l.via, reason });
      setFlagged(new Set([...flagged, key(l)]));
      setOpen(null);
      setReason("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="small" data-testid="graph-entities">
        {data.entities.length === 0
          ? "This transaction carries no device, IP address or beneficiary to link on."
          : data.entities.map((e) => (
              <div key={e.node}>
                {e.kind[0].toUpperCase() + e.kind.slice(1)} <span className="mono">{e.node}</span>:{" "}
                {e.other_customers === 0 ? "not used by anyone else" : `used by ${e.other_customers} other customer${e.other_customers === 1 ? "" : "s"}`}
                {e.hub && <> <Pill tone="grey" title="Used by so many customers that it says little about any one of them">widely shared</Pill></>}
              </div>
            ))}
      </div>
      {data.links.length === 0 ? (
        <div className="small pad-top" data-testid="graph-none">
          No other customer is linked to this case through its device, IP address or beneficiary.
        </div>
      ) : (
        <>
          <div className="small pad-top" data-testid="graph-group">
            Connected group of {data.group.size} customers ({data.group.links} link{data.group.links === 1 ? "" : "s"}
            {data.group.fraud_linked_members > 0 ? `; ${data.group.fraud_linked_members} with a confirmed fraud known at the time` : ""}).
          </div>
          <table className="queue" aria-label="Linked customers">
            <thead><tr><th>Customer</th><th>Linked through</th><th>Confidence</th><th>Seen</th><th>Last seen</th><th /></tr></thead>
            <tbody>
              {data.links.map((l) => (
                <tr key={key(l)} data-testid="graph-link">
                  <td className="mono">{l.other_customer}{l.fraud_known && <> <Pill tone="red">confirmed fraud known</Pill></>}</td>
                  <td className="small">{l.via_kind} <span className="mono">{l.via}</span></td>
                  <td>{pct0(l.confidence)}{(l.flagged_wrong || flagged.has(key(l))) && <> <Pill tone="grey">flagged wrong</Pill></>}</td>
                  <td>{l.observations}×</td>
                  <td className="tiny">{formatTime(l.last_seen)}</td>
                  <td className="right">
                    {canFlag && !l.flagged_wrong && !flagged.has(key(l)) && (
                      <button type="button" className="btn ghost" onClick={() => { setOpen(key(l)); setReason(""); setError(null); }}>Flag as wrong</button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {data.links_total > data.links.length && (
            <div className="tiny muted">Showing the {data.links.length} strongest of {data.links_total} links.</div>
          )}
          {open && (() => {
            const l = data.links.find((x) => key(x) === open);
            return l ? (
              <div className="filters filters-row">
                <label className="field-wide">Why this link is wrong
                  <input aria-label="Reason for flagging" value={reason} onChange={(e) => setReason(e.target.value)} />
                </label>
                <button type="button" className="btn primary" disabled={busy || !reason.trim()} onClick={() => void submit(l)}>Flag link</button>
                <button type="button" className="btn ghost" onClick={() => setOpen(null)}>Cancel</button>
              </div>
            ) : null;
          })()}
        </>
      )}
      {error && <div className="error" role="alert">{error}</div>}
      {data.reliability_note && <div className="tiny pad-top" role="note" data-testid="graph-reliability">{data.reliability_note}</div>}
      <div className="tiny muted pad-top">As the graph stood at {formatTime(data.as_of)}. A flagged link counts for less on later cases and is never deleted.</div>
      <div className="tiny muted pad-top" data-testid="graph-note">{data.note}</div>
    </>
  );
}
