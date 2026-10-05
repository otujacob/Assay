/**
 * Browser sign-in with an OpenID Connect provider: authorization code with PKCE (FR-42). The web app is a PUBLIC client, so
 * there is no client secret; the code is useless without the one-time verifier that stays in this tab (RFC 7636).
 *
 * What the browser does and does not check. It checks `state` (so a callback it did not start is refused) and the `nonce` in
 * the ID token (so a replayed token is refused). It does NOT check the token's signature, issuer, audience, expiry, MFA or
 * roles: the API does, on every request, and the token is only ever treated as an opaque bearer here. Nothing in the browser
 * decides what a person may do.
 *
 * Where the token lives: in memory, and in sessionStorage so a reload does not sign the person out. sessionStorage is
 * per-tab and cleared when the tab closes, but it can be read by any script on the page, so the app must not load third-party
 * script, and the token's lifetime is kept short at the provider (the API also refuses a token whose `iat` is too old).
 */

export interface AuthConfig {
  enabled: boolean;
  client_id?: string;
  authorization_endpoint?: string;
  token_endpoint?: string;
  scope?: string;
  redirect_uri?: string;
  end_session_endpoint?: string;
  audience?: string;
  token_use?: "access_token" | "id_token";
}

export interface Session {
  token: string;
  expiresAt: number; // ms since the epoch
  name: string | null;
  sub: string | null;
}

export class SignInError extends Error {
  constructor(message: string, public code = "sign_in_failed") {
    super(message);
  }
}

const PENDING = "assay.oidc.pending";
const SESSION = "assay.oidc.session";
let memory: Session | null = null;

const store = {
  get: (k: string): string | null => {
    try { return sessionStorage.getItem(k); } catch { return null; }
  },
  set: (k: string, v: string): void => {
    try { sessionStorage.setItem(k, v); } catch { /* storage unavailable: the memory copy still works */ }
  },
  del: (k: string): void => {
    try { sessionStorage.removeItem(k); } catch { /* storage unavailable */ }
  },
};

/** Browser navigation, in one place so tests can see where a sign-in or sign-out would send the person. */
export const nav = { to: (url: string): void => window.location.assign(url) };

export const b64url = (bytes: Uint8Array): string =>
  btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

/** 32 random bytes by default, base64url: 43 characters, inside the 43 to 128 RFC 7636 allows for a verifier. */
export const randomString = (bytes = 32): string => b64url(crypto.getRandomValues(new Uint8Array(bytes)));

/** The S256 code challenge for a verifier. */
export async function challengeFor(verifier: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
  return b64url(new Uint8Array(digest));
}

/** The provider settings the server publishes. If they cannot be read, sign-in is treated as off (demo mode). */
export async function fetchAuthConfig(): Promise<AuthConfig> {
  try {
    const res = await fetch("/api/auth/config");
    if (!res.ok) return { enabled: false };
    const cfg = (await res.json()) as AuthConfig;
    return cfg && cfg.enabled && cfg.client_id && cfg.authorization_endpoint && cfg.token_endpoint ? cfg : { enabled: false };
  } catch {
    return { enabled: false };
  }
}

export const redirectUri = (cfg: AuthConfig): string => cfg.redirect_uri ?? `${window.location.origin}${window.location.pathname}`;

/** The authorization URL, and what must be remembered to finish the sign-in. */
export async function buildAuthorizeUrl(cfg: AuthConfig): Promise<{ url: string; pending: { verifier: string; state: string; nonce: string } }> {
  const pending = { verifier: randomString(64), state: randomString(), nonce: randomString() };
  const q = new URLSearchParams({
    response_type: "code",
    client_id: cfg.client_id!,
    redirect_uri: redirectUri(cfg),
    scope: cfg.scope ?? "openid",
    state: pending.state,
    nonce: pending.nonce,
    code_challenge: await challengeFor(pending.verifier),
    code_challenge_method: "S256",
  });
  if (cfg.audience) q.set("audience", cfg.audience);
  return { url: `${cfg.authorization_endpoint}${cfg.authorization_endpoint!.includes("?") ? "&" : "?"}${q}`, pending };
}

export async function beginLogin(cfg: AuthConfig): Promise<void> {
  const { url, pending } = await buildAuthorizeUrl(cfg);
  store.set(PENDING, JSON.stringify(pending));
  nav.to(url);
}

export type Callback = { code: string; state: string } | { error: string; description: string } | null;

/** What the provider sent back, or null if this URL is not a sign-in callback. */
export function parseCallback(search: string): Callback {
  const p = new URLSearchParams(search);
  if (p.get("error")) return { error: p.get("error")!, description: p.get("error_description") ?? "" };
  const code = p.get("code"), state = p.get("state");
  return code && state ? { code, state } : null;
}

function claimsOf(jwt: string | undefined): Record<string, unknown> {
  try {
    const part = jwt?.split(".")[1];
    if (!part) return {};
    const json = atob(part.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(part.length / 4) * 4, "="));
    return JSON.parse(decodeURIComponent(escape(json))) as Record<string, unknown>;
  } catch {
    return {};
  }
}

/** Exchange the code for a token and keep the session. Throws SignInError with a message a person can act on. */
export async function completeLogin(cfg: AuthConfig, search: string, now: () => number = Date.now): Promise<Session> {
  const cb = parseCallback(search);
  if (cb === null) throw new SignInError("This is not a sign-in response.", "not_a_callback");
  if ("error" in cb) throw new SignInError(`The identity provider refused the sign-in: ${cb.description || cb.error}`, "provider_error");
  const raw = store.get(PENDING);
  store.del(PENDING); // a sign-in attempt can be completed once
  const pending = raw ? (JSON.parse(raw) as { verifier: string; state: string; nonce: string }) : null;
  if (!pending || pending.state !== cb.state) throw new SignInError("The sign-in response did not match a sign-in started here. Please try again.", "state_mismatch");

  let res: Response;
  try {
    res = await fetch(cfg.token_endpoint!, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded", Accept: "application/json" },
      body: new URLSearchParams({
        grant_type: "authorization_code", code: cb.code, redirect_uri: redirectUri(cfg), client_id: cfg.client_id!, code_verifier: pending.verifier,
      }),
    });
  } catch {
    throw new SignInError("Could not reach the identity provider to finish signing in.", "network");
  }
  const body = (await res.json().catch(() => ({}))) as { access_token?: string; id_token?: string; expires_in?: number; error?: string; error_description?: string };
  if (!res.ok) throw new SignInError(`The identity provider did not accept the sign-in${body.error_description ? `: ${body.error_description}` : "."}`, body.error ?? "token_refused");

  const id = claimsOf(body.id_token);
  if (body.id_token && id.nonce !== pending.nonce) throw new SignInError("The sign-in response was not for this sign-in. Please try again.", "nonce_mismatch");
  const token = cfg.token_use === "id_token" ? body.id_token : body.access_token;
  if (!token) throw new SignInError(`The identity provider did not return an ${cfg.token_use === "id_token" ? "ID" : "access"} token.`, "no_token");
  const exp = typeof claimsOf(token).exp === "number" ? (claimsOf(token).exp as number) * 1000 : undefined;
  const expiresAt = typeof body.expires_in === "number" ? now() + body.expires_in * 1000 : (exp ?? now() + 3600_000);
  const session: Session = {
    token, expiresAt,
    name: (id.name ?? id.preferred_username ?? null) as string | null,
    sub: (id.sub ?? null) as string | null,
  };
  memory = session;
  store.set(SESSION, JSON.stringify(session));
  return session;
}

/** The current session, or null if there is none or it has run out. */
export function getSession(now: () => number = Date.now): Session | null {
  if (!memory) {
    const raw = store.get(SESSION);
    if (raw) {
      try { memory = JSON.parse(raw) as Session; } catch { memory = null; }
    }
  }
  if (memory && memory.expiresAt <= now()) clearSession();
  return memory;
}

export const getToken = (): string | null => getSession()?.token ?? null;

export function clearSession(): void {
  memory = null;
  store.del(SESSION);
  store.del(PENDING);
}

/** Where to send the browser to end the provider's own session as well, if it has a logout endpoint. */
export function signOutUrl(cfg: AuthConfig): string | null {
  if (!cfg.end_session_endpoint) return null;
  const q = new URLSearchParams({ client_id: cfg.client_id!, post_logout_redirect_uri: redirectUri(cfg) });
  return `${cfg.end_session_endpoint}${cfg.end_session_endpoint.includes("?") ? "&" : "?"}${q}`;
}
