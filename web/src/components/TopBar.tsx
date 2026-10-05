import type { Dashboard, DemoUser } from "../types";

export function TopBar({
  users, user, onUser, search, onSearch, dash, sso,
}: {
  users: DemoUser[];
  user: string;
  onUser: (key: string) => void;
  search: string;
  onSearch: (s: string) => void;
  dash: Dashboard | null;
  /** Set when a person has signed in with the institution's provider: no demo user switch, and a way to sign out. */
  sso?: { label: string; tenant: string; onSignOut: () => void };
}) {
  const drift = dash?.model_drift.status;
  const dq = dash?.data_quality.status;
  return (
    <header className="topbar">
      <div className="brand">
        <span className="logo" aria-hidden="true">◈</span>
        <span className="brand-name">Assay</span>
        <span className="brand-sub">Adaptive Trust Calibration Engine™</span>
      </div>
      {sso
        ? <div className="tenant" title="Your organisation">{sso.tenant}</div>
        : <div className="tenant" title="Synthetic data: this is a demonstration">Synthetic Bank (demo)</div>}
      <span className="pill pill-green mode" title="Automation level 0: Assay recommends, people decide (FR-23)">● Recommend-Only Mode</span>
      <input className="global-search" aria-label="Search by transaction ID" placeholder="Search by Transaction ID…"
             value={search} onChange={(e) => onSearch(e.target.value)} />
      <div className="status-pills">
        <div className={`status ${drift === "alarm" ? "warn" : "ok"}`}>
          <span className="status-dot" /> <span><span className="tiny muted">Model Drift</span><br />{drift === "alarm" ? "Alarm" : drift ? "Stable" : "n/a"}</span>
        </div>
        <div className={`status ${dq === "degraded" ? "warn" : "ok"}`}>
          <span className="status-dot" /> <span><span className="tiny muted">Data Quality</span><br />{dq === "degraded" ? "Degraded" : dq ? "Healthy" : "n/a"}</span>
        </div>
      </div>
      {sso ? (
        <div className="user-switch" data-testid="signed-in-user">
          <span className="tiny muted">Signed in as</span>
          <span>{sso.label} <button type="button" className="btn ghost" onClick={sso.onSignOut}>Sign out</button></span>
        </div>
      ) : (
        <label className="user-switch" title="Demo only: real deployments use single sign-on">
          <span className="tiny muted">Demo user</span>
          <select aria-label="Demo user" value={user} onChange={(e) => onUser(e.target.value)}>
            {users.map((u) => <option key={u.key} value={u.key}>{u.label}</option>)}
          </select>
        </label>
      )}
    </header>
  );
}
