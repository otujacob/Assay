import { useCallback, useEffect, useState } from "react";
import { api, type CandidateStep } from "../api";
import type { Candidate, CandidateState, FeedbackPool, Gate, LearningStatus } from "../types";
import { formatTime } from "../lib/format";
import { Pill, type Tone } from "./Pill";

const STATE: Record<CandidateState, { tone: Tone; label: string }> = {
  validated: { tone: "blue", label: "Validated" },
  failed: { tone: "red", label: "Failed a gate" },
  shadow: { tone: "purple", label: "In shadow" },
  approved: { tone: "amber", label: "Approved" },
  canary: { tone: "amber", label: "Canary" },
  champion: { tone: "green", label: "Champion" },
  rolled_back: { tone: "grey", label: "Rolled back" },
  rejected: { tone: "grey", label: "Rejected" },
};

const GATE_TONE: Record<Gate["status"], Tone> = {
  pass: "green", fail: "red", not_evaluated: "amber", review_required: "amber", pending: "grey",
};
const GATE_WORD: Record<Gate["status"], string> = {
  pass: "Pass", fail: "Fail", not_evaluated: "Could not judge", review_required: "Needs review", pending: "Later step",
};

/** G7 and G9 are judged after the report is stored (at creation they are "later steps"). Once the history shows the
 *  candidate was approved, both were met, so show them as met; the evidence is in the history below. */
const PAST_APPROVAL: ReadonlySet<CandidateState> = new Set(["approved", "canary", "champion"]);
export function shownGates(c: Candidate): Gate[] {
  if (!PAST_APPROVAL.has(c.state)) return c.gates.gates;
  return c.gates.gates.map((g) =>
    (g.id === "G7" || g.id === "G9") && g.status === "pending" ? { ...g, status: "pass", detail: "recorded in the history below" } : g);
}

const pct = (x: number | null | undefined) => (x === null || x === undefined ? "n/a" : `${Math.round(x * 100)}%`);

/** What a person has to do next, and the controls for it. The server decides what is allowed; this only
 *  offers what the role and the state can do, and shows the server's refusal if there is one. */
function Controls({ c, me, roles, busy, run }: {
  c: Candidate; me: string; roles: ReadonlySet<string>; busy: boolean;
  run: (step: CandidateStep, body?: Record<string, unknown>) => void;
}) {
  const isApprover = roles.has("approver");
  const canStart = roles.has("admin") || isApprover;
  const [rationale, setRationale] = useState("");
  const [reviewed, setReviewed] = useState(false);
  const [waived, setWaived] = useState<string[]>([]);
  const [share, setShare] = useState("0.1");
  const [reason, setReason] = useState("");
  const cannotJudge = c.gates.gates.filter((g) => g.status === "not_evaluated").map((g) => g.id);
  const own = c.created_by === me;

  const reasonBox = (label: string, step: CandidateStep) => (
    <div className="filters filters-row">
      <label className="field-wide">{label}
        <input aria-label={`${label} (reason)`} value={reason} onChange={(e) => setReason(e.target.value)} />
      </label>
      <button type="button" className="btn ghost" disabled={busy || !reason.trim()} onClick={() => run(step, { reason })}>{label}</button>
    </div>
  );

  return (
    <div data-testid="candidate-controls">
      {c.state === "validated" && canStart && (
        <div className="actions pad">
          <button type="button" className="btn primary" disabled={busy} onClick={() => run("shadow")}>Start shadow</button>
          <span className="tiny muted"> The candidate will score live traffic and take no action.</span>
        </div>
      )}
      {c.state === "shadow" && isApprover && (
        <div className="pad">
          {own && <div className="tiny muted" role="note">You created this candidate, so someone else must approve it.</div>}
          <label className="tiny muted">Why you are approving it
            <input aria-label="Approval rationale" style={{ width: "100%" }} value={rationale} onChange={(e) => setRationale(e.target.value)} />
          </label>
          {cannotJudge.length > 0 && (
            <div className="tiny" role="group" aria-label="Gates that could not be judged">
              These gates could not be judged. Waive each one you accept going without:{" "}
              {cannotJudge.map((id) => (
                <label key={id} className="tiny" style={{ marginRight: 10 }}>
                  <input type="checkbox" checked={waived.includes(id)} onChange={(e) =>
                    setWaived(e.target.checked ? [...waived, id] : waived.filter((w) => w !== id))} /> {id}
                </label>
              ))}
            </div>
          )}
          <label className="tiny">
            <input type="checkbox" checked={reviewed} onChange={(e) => setReviewed(e.target.checked)} /> I reviewed the results by segment (G6)
          </label>
          <div className="actions">
            <button type="button" className="btn primary" disabled={busy || own || !rationale.trim() || !reviewed}
                    title={own ? "You created this candidate" : undefined}
                    onClick={() => run("approve", { rationale, reviewed_segments: reviewed, waived })}>Approve</button>
          </div>
        </div>
      )}
      {c.state === "approved" && isApprover && (
        <div className="filters filters-row">
          <label>Share of traffic the candidate decides
            <input aria-label="Canary share" inputMode="decimal" value={share} onChange={(e) => setShare(e.target.value)} />
          </label>
          <button type="button" className="btn primary" disabled={busy || !(Number(share) > 0)}
                  onClick={() => run("canary", { share: Number(share) })}>Start canary</button>
        </div>
      )}
      {c.state === "canary" && isApprover && (
        <div className="actions pad">
          <button type="button" className="btn primary" disabled={busy} onClick={() => run("promote")}>Promote to champion</button>
        </div>
      )}
      {(c.state === "canary" || c.state === "champion") && isApprover && reasonBox("Roll back", "rollback")}
      {(c.state === "validated" || c.state === "failed" || c.state === "shadow" || c.state === "approved" || c.state === "canary") && canStart
        && reasonBox("Reject", "reject")}
    </div>
  );
}

function CandidateCard({ c, me, roles, onChanged }: {
  c: Candidate; me: string; roles: ReadonlySet<string>; onChanged: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const s = STATE[c.state];
  const run = (step: CandidateStep, body: Record<string, unknown> = {}) => {
    setBusy(true);
    api.candidateStep(c.candidate_id, step, body)
      .then(() => { setError(null); return onChanged(); })
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };
  return (
    <div className="pad" data-testid={`candidate-${c.candidate_id}`}>
      <div>
        <span className="mono">{c.candidate_id}</span> <Pill tone={s.tone}>{s.label}</Pill>{" "}
        <span className="tiny muted">created by {c.created_by}, compared with {c.base_bundle_id}</span>
      </div>
      <div className="tiny muted">
        Trained with {c.training.extra_labels_used ?? 0} accepted analyst labels
        {c.training.feedback_used === false ? " (feedback not used)" : ""}.
      </div>
      <table className="queue" aria-label={`Gates for ${c.candidate_id}`}>
        <tbody>
          {shownGates(c).map((g) => (
            <tr key={g.id}>
              <td className="mono">{g.id}</td><td>{g.name}</td>
              <td><Pill tone={GATE_TONE[g.status]}>{GATE_WORD[g.status]}</Pill></td>
              <td className="small" style={{ whiteSpace: "normal" }}>{g.detail}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {c.shadow && (
        <div className="tiny" data-testid="shadow-report">
          Shadow: {c.shadow.detail}.
          {c.shadow.agreement !== undefined && <> Agrees with the champion on {pct(c.shadow.agreement)} of cases.</>}
          {c.shadow.alarms.length > 0 && <> <b>Alarm:</b> {c.shadow.alarms.join("; ")}.</>}
        </div>
      )}
      {error && <div className="banner-error" role="alert">{error}</div>}
      <Controls c={c} me={me} roles={roles} busy={busy} run={run} />
      <details>
        <summary className="tiny muted">History ({c.events.length})</summary>
        <ul className="tiny">
          {c.events.map((e, i) => (
            <li key={i}>{formatTime(e.at)} · {e.kind.replace(/_/g, " ")} · {e.actor}
              {e.reason ? ` · ${e.reason}` : ""}{e.rationale ? ` · ${e.rationale}` : ""}{e.share ? ` · ${pct(e.share)} of traffic` : ""}</li>
          ))}
        </ul>
      </details>
    </div>
  );
}

/** Candidate models and the feedback behind them (PRD 11, 12). Nothing here happens by itself: each step is a person's. */
export function ModelUpdates({ userKey, roles }: { userKey: string; roles: ReadonlySet<string> }) {
  const [status, setStatus] = useState<LearningStatus | null>(null);
  const [pool, setPool] = useState<FeedbackPool | null>(null);
  const [items, setItems] = useState<Candidate[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [st, pl, cs] = await Promise.all([api.learningStatus(), api.feedbackPool(), api.candidates()]);
      setStatus(st); setPool(pl); setItems(cs.items); setError(null);
    } catch (e) {
      setError(`Could not load model updates: ${(e as Error).message}`);
    }
  }, []);
  useEffect(() => { void load(); }, [load]);

  return (
    <div className="panel" data-testid="model-updates">
      <div className="panel-head">
        <div className="panel-title">Model Updates</div>
        <button type="button" className="btn ghost" onClick={() => void load()}>Refresh</button>
      </div>
      <div className="gov-note tiny">
        Assay never retrains or promotes a model by itself. A candidate is trained from verified outcomes plus feedback that passed a
        quality check, must pass the validation gates, run in shadow, be approved by a <b>different</b> person, take a share of live
        traffic as a canary, and only then be promoted. The previous champion stays deployable and a rollback restores it.
      </div>
      {error && <div className="banner-error" role="alert">{error}</div>}
      {status && (
        <div className="pad small" data-testid="learning-status">
          Champion <span className="mono">{status.champion ?? "none"}</span>
          {status.shadow && <> · in shadow <span className="mono">{status.shadow}</span></>}
          {status.canary && <> · canary <span className="mono">{status.canary.candidate}</span> at {pct(status.canary.share)} of traffic</>}
        </div>
      )}
      {pool && (
        <div className="pad small" data-testid="feedback-pool">
          <b>Analyst feedback.</b> {pool.accepted} accepted, {pool.rejected} rejected, {pool.deferred} deferred
          {pool.acceptance_rate !== null && <> · {pct(pool.acceptance_rate)} of unverified decisions accepted</>}.
          {Object.keys(pool.flagged).length > 0 && (
            <div className="tiny" role="note">
              Quarantined for review: {Object.entries(pool.flagged).map(([a, why]) => `${a} (${why.join("; ")})`).join(", ")}.
            </div>
          )}
          <div className="tiny muted">{pool.note}</div>
        </div>
      )}
      {items && items.length === 0 && (
        <div className="empty pad">
          No candidate models yet. A candidate is created with <span className="mono">python -m assay.learning create</span>.
        </div>
      )}
      {(items ?? []).map((c) => (
        <CandidateCard key={c.candidate_id} c={c} me={`u:${userKey}`} roles={roles} onChanged={load} />
      ))}
    </div>
  );
}
