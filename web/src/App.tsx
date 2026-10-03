import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api, getUser, setUser } from "./api";
import { AuditLog } from "./components/AuditLog";
import { CaseDetails } from "./components/CaseDetails";
import { Governance } from "./components/Governance";
import { KpiCards } from "./components/KpiCards";
import { EMPTY_FILTERS, ReviewQueue, type QueueFilterState } from "./components/ReviewQueue";
import { TopBar } from "./components/TopBar";
import type {
  ActionRequest, ActionResponse, BundleInfo, CaseView, Dashboard, DemoUser, QueueResponse, ValidationReport,
} from "./types";

const ok = <T,>(r: PromiseSettledResult<T>): T | null => (r.status === "fulfilled" ? r.value : null);

export default function App() {
  const [users, setUsers] = useState<DemoUser[]>([]);
  const [user, setUserState] = useState(getUser());
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
    api.users().then(setUsers).catch((e: Error) => setError(`Cannot reach the demo server: ${e.message}`));
  }, []);

  const loadQueue = useCallback(async (f: QueueFilterState) => {
    try {
      const q = await api.queue({
        risk_band: f.risk, trust_state: f.trust, reason_code: f.reason, search: f.search,
      });
      fetchedAt.current = Date.now();
      setElapsed(0);
      setQueue(q);
      setError(null);
    } catch (e) {
      setQueue(null);
      setError(e instanceof ApiError && e.status === 403 ? null : `Could not load the queue: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadSide = useCallback(async () => {
    const [d, r, b] = await Promise.allSettled([api.dashboard(), api.reports(), api.bundles()]);
    setDash(ok(d));
    setReport(ok(r)?.[0] ?? null);
    setBundle(ok(b)?.find((x) => x.champion) ?? ok(b)?.[0] ?? null);
  }, []);

  const loadCase = useCallback(async (id: string) => {
    try {
      setView(await api.getCase(id));
    } catch (e) {
      setView(null);
      setError(`Could not load the case: ${(e as Error).message}`);
    }
  }, []);

  // Reload everything when the demo user changes.
  useEffect(() => {
    if (!users.length) return;
    setLoading(true);
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
      <div className="demo-banner">
        DEMO: synthetic data and a dev-only signing proxy. Not for production: real deployments use single sign-on.
      </div>
      <TopBar users={users} user={user} onUser={switchUser} search={filters.search}
              onSearch={(s) => setFilters({ ...filters, search: s })} dash={dash} />
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
