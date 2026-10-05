import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api, getUser, setUser } from "./api";
import { AuditLog } from "./components/AuditLog";
import { CaseDetails } from "./components/CaseDetails";
import { Governance } from "./components/Governance";
import { ModelUpdates } from "./components/ModelUpdates";
import { PolicyPanel } from "./components/PolicyPanel";
import { KpiCards } from "./components/KpiCards";
import { EMPTY_FILTERS, ReviewQueue, type QueueFilterState } from "./components/ReviewQueue";
import { TopBar } from "./components/TopBar";
import type {
  ActionRequest, ActionResponse, BundleInfo, CaseView, Dashboard, DemoUser, QueueResponse, ValidationReport,
} from "./types";

const ok = <T,>(r: PromiseSettledResult<T>): T | null => (r.status === "fulfilled" ? r.value : null);

// A response belongs to the user who asked. If the demo user has changed since, it is dropped, so a
// slow answer for the previous user cannot fill the new user's screen or open a case they may not see.
const stale = (asked: string): boolean => getUser() !== asked;

export interface SsoProps {
  user: DemoUser;  // who signed in, and the roles the server says they hold
  tenant: string;
  onSignOut: () => void;
}

export default function App({ sso }: { sso?: SsoProps } = {}) {
  const [users, setUsers] = useState<DemoUser[]>(sso ? [sso.user] : []);
  const [user, setUserState] = useState(sso ? sso.user.key : getUser());
  const [queue, setQueue] = useState<QueueResponse | null>(null);
  const [filters, setFilters] = useState<QueueFilterState>(EMPTY_FILTERS);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<CaseView | null>(null);
  const [dash, setDash] = useState<Dashboard | null>(null);
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [bundle, setBundle] = useState<BundleInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const fetchedAt = useRef(Date.now());
  const autoSelected = useRef(false);

  const roles = useMemo(() => new Set(users.find((u) => u.key === user)?.roles ?? []), [users, user]);
  const canAct = roles.has("analyst") || roles.has("senior_analyst");
  const canAdjudicate = roles.has("senior_analyst");
  const canQueue = canAct || roles.has("manager");

  // SLA timers count down between fetches.
  useEffect(() => {
    const t = setInterval(() => setElapsed((Date.now() - fetchedAt.current) / 1000), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (sso) return; // a signed-in person is the only "user"; there is no demo user list
    api.users().then(setUsers).catch((e: Error) => setError(`Cannot reach the demo server: ${e.message}`));
  }, [sso]);

  const loadQueue = useCallback(async (f: QueueFilterState) => {
    const asked = getUser();
    try {
      const q = await api.queue({
        risk_band: f.risk, trust_state: f.trust, reason_code: f.reason, search: f.search,
      });
      if (stale(asked)) return;
      fetchedAt.current = Date.now();
      setElapsed(0);
      setQueue(q);
      setError(null);
    } catch (e) {
      if (stale(asked)) return;
      setQueue(null);
      setError(e instanceof ApiError && e.status === 403 ? null : `Could not load the queue: ${(e as Error).message}`);
    } finally {
      if (!stale(asked)) setLoading(false);
    }
  }, []);

  const loadSide = useCallback(async () => {
    const asked = getUser();
    const [d, r, b] = await Promise.allSettled([api.dashboard(), api.reports(), api.bundles()]);
    if (stale(asked)) return;
    setDash(ok(d));
    setReport(ok(r)?.[0] ?? null);
    setBundle(ok(b)?.find((x) => x.champion) ?? ok(b)?.[0] ?? null);
  }, []);

  const loadCase = useCallback(async (id: string) => {
    const asked = getUser();
    try {
      const v = await api.getCase(id);
      if (!stale(asked)) setView(v);
    } catch (e) {
      if (stale(asked)) return;
      setView(null);
      setError(`Could not load the case: ${(e as Error).message}`);
    }
  }, []);

  // Reload everything when the demo user changes.
  useEffect(() => {
    if (!users.length) return;
    setLoading(true);
    setError(null);
    setQueue(null); // the previous user's queue must not survive: auto-select would open its first case as this user
    setSelectedId(null);
    setView(null);
    autoSelected.current = false;
    void loadQueue(filters);
    void loadSide();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user, users.length]);

  // Filters change the query (server-side, so redaction is applied there). Search is debounced.
  useEffect(() => {
    if (!users.length) return;
    const t = setTimeout(() => void loadQueue(filters), filters.search ? 250 : 0);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters]);

  useEffect(() => {
    if (queue && !autoSelected.current && queue.items.length > 0 && !selectedId) {
      autoSelected.current = true;
      setSelectedId(queue.items[0].decision_id);
    }
  }, [queue, selectedId]);

  useEffect(() => {
    if (selectedId) void loadCase(selectedId);
  }, [selectedId, loadCase]);

  const refresh = async () => {
    await Promise.all([loadQueue(filters), loadSide(), selectedId ? loadCase(selectedId) : Promise.resolve()]);
  };

  const onAct = async (body: ActionRequest): Promise<ActionResponse> => {
    if (!selectedId) throw new Error("no case selected");
    const res = await api.act(selectedId, body);
    await refresh();
    return res;
  };

  const onAdjudicate = async (final: "approve" | "block", rationale: string) => {
    if (!selectedId) return;
    await api.adjudicate(selectedId, { final_decision: final, rationale });
    await refresh();
  };

  const switchUser = (key: string) => {
    setUser(key);
    setUserState(key);
  };

  return (
    <div className="app">
      {!sso && (
        <div className="demo-banner">
          DEMO: synthetic data and a dev-only signing proxy. Not for production: real deployments use single sign-on.
        </div>
      )}
      <TopBar users={users} user={user} onUser={switchUser} search={filters.search}
              onSearch={(s) => setFilters({ ...filters, search: s })} dash={dash}
              sso={sso ? { label: sso.user.label, tenant: sso.tenant, onSignOut: sso.onSignOut } : undefined} />
      {error && <div className="banner-error" role="alert">{error}</div>}
      <div className="page-title">
        <h1>AI Decision Intelligence Dashboard</h1>
        <p className="muted">Review, investigate and record human decisions on transactions where the model should not be relied on alone.</p>
      </div>
      <KpiCards dash={dash} elapsedSeconds={elapsed} />
      <div className="main-grid">
        <div className="left-col">
          {canQueue ? (
            <ReviewQueue items={queue?.items ?? []} selectedId={selectedId} onSelect={setSelectedId}
                         filters={filters} onFilters={setFilters} elapsedSeconds={elapsed}
                         total={queue?.matching ?? 0} loading={loading} />
          ) : (
            <div className="panel pad muted">Your role does not include the review queue.</div>
          )}
          <Governance report={report} bundle={bundle} />
          {(roles.has("admin") || roles.has("approver") || roles.has("auditor")) && (
            <PolicyPanel userKey={user} roles={roles} />
          )}
          {(roles.has("admin") || roles.has("approver") || roles.has("auditor")) && (
            <ModelUpdates userKey={user} roles={roles} />
          )}
          {roles.has("auditor") && <AuditLog />}
        </div>
        <div className="right-col">
          <div className="panel">
            <div className="tabs" role="tablist">
              <button role="tab" aria-selected className="tab on">Case Details</button>
            </div>
            {view ? (
              <CaseDetails view={view} canAct={canAct} canAdjudicate={canAdjudicate}
                           onAct={onAct} onAdjudicate={onAdjudicate} onBack={() => { setSelectedId(null); setView(null); }} />
            ) : (
              <div className="empty pad">{selectedId ? "Loading case…" : "Select a case from the queue."}</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
