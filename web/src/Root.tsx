import { useCallback, useEffect, useState } from "react";
import App from "./App";
import { ApiError, api, setUnauthorizedHandler, setUser } from "./api";
import {
  type AuthConfig, SignInError, beginLogin, clearSession, completeLogin, fetchAuthConfig, getSession, nav, parseCallback, signOutUrl,
} from "./auth/oidc";
import type { DemoUser } from "./types";

type State =
  | { kind: "loading" }
  | { kind: "demo" }
  | { kind: "signin"; cfg: AuthConfig; message?: string }
  | { kind: "noaccess"; cfg: AuthConfig; who: string }
  | { kind: "signed"; cfg: AuthConfig; user: DemoUser; tenant: string };

/** Take the sign-in parameters out of the address bar, so a reload does not try to use a code twice. */
const cleanUrl = () => window.history.replaceState(null, "", window.location.pathname);

/**
 * Chooses how the app starts. If the server publishes single sign-on settings, a person must sign in with the institution's
 * provider; otherwise this is the demo, which uses its signing proxy. The server decides what a signed-in person may do (roles
 * come from /v1/me, which reads the verified token): nothing in the browser grants access.
 */
export default function Root() {
  const [state, setState] = useState<State>({ kind: "loading" });

  const signedOut = useCallback((cfg: AuthConfig, message?: string) => {
    clearSession();
    setState({ kind: "signin", cfg, message });
  }, []);

  useEffect(() => {
    let live = true;
    (async () => {
      const cfg = await fetchAuthConfig();
      if (!live) return;
      if (!cfg.enabled) { setState({ kind: "demo" }); return; }
      const cb = parseCallback(window.location.search);
      if (cb) {
        try {
          await completeLogin(cfg, window.location.search);
        } catch (e) {
          cleanUrl();
          if (live) signedOut(cfg, (e as SignInError).message);
          return;
        }
        cleanUrl();
      }
      if (!getSession()) { if (live) setState({ kind: "signin", cfg }); return; }
      try {
        const me = await api.me();
        if (!live) return;
        const session = getSession();
        if (me.roles.length === 0) { setState({ kind: "noaccess", cfg, who: session?.name ?? me.subject }); return; }
        setUser(me.subject); // the app keeps its "who asked" guard keyed on this
        setState({ kind: "signed", cfg, tenant: me.tenant, user: { key: me.subject, label: session?.name ?? "Signed in", roles: me.roles } });
      } catch (e) {
        if (!live) return;
        signedOut(cfg, e instanceof ApiError && e.code === "mfa_required"
          ? "Your sign-in did not use multi-factor authentication, which is required. Please sign in again with MFA."
          : "Your session could not be confirmed. Please sign in again.");
      }
    })();
    return () => { live = false; };
  }, [signedOut]);

  // Once signed in, a token the API stops accepting (it expired, or MFA was missing) ends the session.
  useEffect(() => {
    if (state.kind !== "signed") return;
    const cfg = state.cfg;
    setUnauthorizedHandler((code) => signedOut(cfg, code === "mfa_required"
      ? "Multi-factor authentication is required. Please sign in again with MFA."
      : "Your session has ended. Please sign in again."));
    return () => setUnauthorizedHandler(null);
  }, [state, signedOut]);

  const signOut = (cfg: AuthConfig) => {
    clearSession();
    const url = signOutUrl(cfg);
    if (url) nav.to(url);
    else setState({ kind: "signin", cfg, message: "You are signed out." });
  };

  switch (state.kind) {
    case "loading":
      return <div className="empty pad" role="status">Loading…</div>;
    case "demo":
      return <App />;
    case "signed":
      return <App sso={{ user: state.user, tenant: state.tenant, onSignOut: () => signOut(state.cfg) }} />;
    case "noaccess":
      return (
        <div className="signin" data-testid="no-access">
          <h1>No access</h1>
          <p>{state.who} signed in, but none of your groups gives you a role in Assay. Ask your administrator to add you to a group that does.</p>
          <button type="button" className="btn ghost" onClick={() => signOut(state.cfg)}>Sign out</button>
        </div>
      );
    case "signin":
      return (
        <div className="signin" data-testid="sign-in">
          <h1>Assay</h1>
          <p className="muted">Sign in with your organisation&rsquo;s account to review cases.</p>
          {state.message && <div className="banner-error" role="alert">{state.message}</div>}
          <button type="button" className="btn primary" onClick={() => void beginLogin(state.cfg)}>Sign in</button>
        </div>
      );
  }
}
