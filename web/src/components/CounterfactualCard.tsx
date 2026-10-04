import { useState } from "react";
import { api } from "../api";
import type { CounterfactualItem, CounterfactualView } from "../types";
import { pct } from "../lib/format";
import { Pill } from "./Pill";

const GROUP_LABEL: Record<string, string> = {
  amount: "amount", timing: "time of day", channel: "channel", counterparty: "beneficiary",
  device_location: "device and location", velocity: "recent activity", completeness: "data completeness",
};
const groupName = (g: string) => GROUP_LABEL[g] ?? g;

/** A counterfactual that passed every check and holds up under small changes. */
const solid = (c: CounterfactualItem) => c.valid && (c.robust_share ?? 0) >= 0.8;

/**
 * "What would change the decision?" for an investigator (PRD 7.1). Loaded on request, because the model is
 * asked to re-score hundreds of variations. It describes the MODEL, not cause and effect, and is not advice
 * for a customer (PRD 7.4): the server's caveat is always shown with the result.
 */
export function CounterfactualCard({ decisionId }: { decisionId: string }) {
  const [data, setData] = useState<CounterfactualView | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setBusy(true);
    setError(null);
    try {
      setData(await api.counterfactuals(decisionId));
    } catch (e) {
      setData(null);
      setError(`Could not work out what-ifs: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card" data-testid="counterfactuals">
      <div className="card-title">What would change the decision?</div>
      {!data && (
        <>
          <div className="tiny muted">
            Small, realistic changes that would have made the model decide differently. For investigators: it shows what the
            model leans on, not what caused anything.
          </div>
          <div className="pad-top">
            <button type="button" className="btn ghost" disabled={busy} onClick={() => void load()}>
              {busy ? "Working it out…" : "Show what-ifs"}
            </button>
          </div>
        </>
      )}
      {error && <div className="error" role="alert">{error}</div>}
      {data && <Result data={data} />}
    </div>
  );
}

export function Result({ data }: { data: CounterfactualView }) {
  const good = data.counterfactuals.filter(solid);
  const verb = data.flagged ? "not be flagged" : "be flagged";
  const cm = data.cross_method;
  return (
    <>
      <div className="small">
        The model {data.flagged ? "flagged" : "did not flag"} this at {pct(data.risk, 1)} (threshold {pct(data.threshold, 1)}).
      </div>
      {good.length > 0 ? (
        <ul className="cf-list" data-testid="cf-list">
          {good.map((c, i) => (
            <li key={i} className="small">
              <b>Would {verb} if</b> {c.changes.map((ch) => ch.text).join(" and ")}
              <span className="muted tiny"> · risk {pct(c.risk_before, 1)} {"→"} {pct(c.risk_after, 1)} · changes {c.groups.map(groupName).join(" and ")}</span>
            </li>
          ))}
        </ul>
      ) : data.counterfactuals.length > 0 ? (
        <div className="small pad-top">
          Some changes were found, but none held up under the checks (they were too fine a margin, or unrealistic). Treat
          this decision as firm.
        </div>
      ) : (
        <div className="small pad-top" data-testid="cf-none">
          No realistic change to the amount, time, channel, beneficiary or device would have changed this call. It is firm.
        </div>
      )}
      <div className="tiny pad-top" data-testid="cf-methods">
        <Pill tone={cm.agreement >= 0.5 ? "green" : "amber"}>Two methods agree {Math.round(cm.agreement * 100)}%</Pill>{" "}
        <span className="muted">
          SHAP names {cm.shap_drivers.map(groupName).join(", ")}; permutation names {cm.permutation_drivers.map(groupName).join(", ")}.
        </span>
        {cm.agreement < 0.5 && (
          <div className="muted">They disagree about what drove this score, so treat the explanation with extra care.</div>
        )}
      </div>
      <div className="tiny muted pad-top" data-testid="cf-note">{data.note}</div>
    </>
  );
}
