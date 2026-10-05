import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  type AuthConfig, SignInError, b64url, buildAuthorizeUrl, challengeFor, clearSession, completeLogin, fetchAuthConfig, getSession, getToken,
  parseCallback, randomString, signOutUrl,
} from "./oidc";

const CFG: AuthConfig = {
  enabled: true, client_id: "assay-web", authorization_endpoint: "https://idp.example/authorize", token_endpoint: "https://idp.example/token",
  scope: "openid profile", end_session_endpoint: "https://idp.example/logout",
};
const jwt = (claims: Record<string, unknown>) => `h.${btoa(JSON.stringify(claims)).replace(/=+$/, "")}.s`;
const stash = (over: Record<string, string> = {}) =>
  sessionStorage.setItem("assay.oidc.pending", JSON.stringify({ verifier: "v".repeat(64), state: "st", nonce: "no", ...over }));
const tokenResponse = (body: unknown, status = 200) => vi.fn(async () => new Response(JSON.stringify(body), { status }));

beforeEach(() => { sessionStorage.clear(); clearSession(); vi.unstubAllGlobals(); });
afterEach(() => vi.unstubAllGlobals());

describe("PKCE", () => {
  it("makes the S256 challenge from the RFC 7636 worked example", async () => {
    expect(await challengeFor("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")).toBe("E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
  });

  it("makes verifiers that are long enough, URL-safe and different every time", () => {
    const a = randomString(), b = randomString(64);
    expect(a).toMatch(/^[A-Za-z0-9_-]{43}$/);
    expect(b.length).toBeGreaterThanOrEqual(43);
    expect(b.length).toBeLessThanOrEqual(128);
    expect(randomString()).not.toBe(a);
    expect(b64url(new Uint8Array([251, 255, 254]))).toBe("-__-"); // + and / become - and _, with no padding
  });

  it("builds an authorization URL for the code flow with PKCE, a state and a nonce, and the audience when there is one", async () => {
    const { url, pending } = await buildAuthorizeUrl({ ...CFG, audience: "assay-api" });
    const u = new URL(url);
    expect(u.origin + u.pathname).toBe("https://idp.example/authorize");
    expect(Object.fromEntries(u.searchParams)).toMatchObject({
      response_type: "code", client_id: "assay-web", scope: "openid profile", code_challenge_method: "S256", audience: "assay-api",
      state: pending.state, nonce: pending.nonce, code_challenge: await challengeFor(pending.verifier),
    });
    expect(u.searchParams.get("redirect_uri")).toBe(`${window.location.origin}${window.location.pathname}`);
    expect(url).not.toContain(pending.verifier); // the verifier never leaves this tab until the token request
    const { url: url2 } = await buildAuthorizeUrl(CFG);
    expect(new URL(url2).searchParams.has("audience")).toBe(false);
  });
});

describe("the published settings", () => {
  it("are used only when sign-in is on and complete; anything else means demo mode", async () => {
    vi.stubGlobal("fetch", tokenResponse(CFG));
    expect((await fetchAuthConfig()).enabled).toBe(true);
    vi.stubGlobal("fetch", tokenResponse({ enabled: false }));
    expect((await fetchAuthConfig()).enabled).toBe(false);
    vi.stubGlobal("fetch", tokenResponse({ enabled: true, client_id: "x" })); // endpoints missing
    expect((await fetchAuthConfig()).enabled).toBe(false);
    vi.stubGlobal("fetch", tokenResponse({}, 404));
    expect((await fetchAuthConfig()).enabled).toBe(false);
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("offline"); }));
    expect((await fetchAuthConfig()).enabled).toBe(false);
  });
});

describe("the callback", () => {
  it("is recognised, or not", () => {
    expect(parseCallback("?code=abc&state=xyz")).toEqual({ code: "abc", state: "xyz" });
    expect(parseCallback("?error=access_denied&error_description=No")).toEqual({ error: "access_denied", description: "No" });
    expect(parseCallback("")).toBeNull();
    expect(parseCallback("?code=abc")).toBeNull(); // no state: not one of ours
  });

  it("is exchanged with the one-time verifier, and gives a session that is kept for the tab", async () => {
    stash();
    const fetchMock = tokenResponse({ access_token: "ACCESS", id_token: jwt({ nonce: "no", name: "Ada", sub: "u1" }), expires_in: 600 });
    vi.stubGlobal("fetch", fetchMock);
    const before = Date.now();
    const s = await completeLogin(CFG, "?code=thecode&state=st");
    expect(s).toMatchObject({ token: "ACCESS", name: "Ada", sub: "u1" });
    expect(s.expiresAt).toBeGreaterThanOrEqual(before + 600_000);
    expect(s.expiresAt).toBeLessThanOrEqual(Date.now() + 600_000);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("https://idp.example/token");
    const body = new URLSearchParams(init.body as string);
    expect(Object.fromEntries(body)).toMatchObject({ grant_type: "authorization_code", code: "thecode", client_id: "assay-web", code_verifier: "v".repeat(64) });
    expect(body.has("client_secret")).toBe(false); // a public client has none
    expect(getToken()).toBe("ACCESS");
    expect(sessionStorage.getItem("assay.oidc.session")).toContain("ACCESS");
    expect(sessionStorage.getItem("assay.oidc.pending")).toBeNull();
  });

  it("can send the ID token as the bearer when the deployment says so", async () => {
    stash();
    const id = jwt({ nonce: "no", exp: 2_000_000_000 });
    vi.stubGlobal("fetch", tokenResponse({ access_token: "ACCESS", id_token: id }));
    const s = await completeLogin({ ...CFG, token_use: "id_token" }, "?code=c&state=st", () => 5);
    expect(s.token).toBe(id);
    expect(s.expiresAt).toBe(2_000_000_000 * 1000); // from the token's own exp when the response gives no lifetime
  });

  it("is refused when the state does not match one we started, and a sign-in can be completed only once", async () => {
    stash();
    vi.stubGlobal("fetch", tokenResponse({ access_token: "A" }));
    await expect(completeLogin(CFG, "?code=c&state=other")).rejects.toMatchObject({ code: "state_mismatch" });
    await expect(completeLogin(CFG, "?code=c&state=st")).rejects.toMatchObject({ code: "state_mismatch" }); // the attempt is gone
    expect(getToken()).toBeNull();
  });

  it("is refused when the ID token carries a different nonce", async () => {
    stash();
    vi.stubGlobal("fetch", tokenResponse({ access_token: "A", id_token: jwt({ nonce: "someone-else" }) }));
    await expect(completeLogin(CFG, "?code=c&state=st")).rejects.toMatchObject({ code: "nonce_mismatch" });
    expect(getToken()).toBeNull();
  });

  it("explains a refusal from the provider, a refused token request, a missing token and an unreachable provider", async () => {
    await expect(completeLogin(CFG, "?error=access_denied&error_description=User+cancelled")).rejects.toThrow(/User cancelled/);
    stash();
    vi.stubGlobal("fetch", tokenResponse({ error: "invalid_grant", error_description: "code expired" }, 400));
    await expect(completeLogin(CFG, "?code=c&state=st")).rejects.toThrow(/code expired/);
    stash();
    vi.stubGlobal("fetch", tokenResponse({ id_token: jwt({ nonce: "no" }) }));
    await expect(completeLogin(CFG, "?code=c&state=st")).rejects.toMatchObject({ code: "no_token" });
    stash();
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("offline"); }));
    await expect(completeLogin(CFG, "?code=c&state=st")).rejects.toMatchObject({ code: "network" });
    await expect(completeLogin(CFG, "")).rejects.toBeInstanceOf(SignInError);
  });
});

describe("the session", () => {
  it("survives a reload within the tab but not past its expiry, and can be cleared", async () => {
    stash();
    vi.stubGlobal("fetch", tokenResponse({ access_token: "A", expires_in: 100 }));
    await completeLogin(CFG, "?code=c&state=st", () => 0);
    expect(getSession(() => 50_000)?.token).toBe("A");
    clearSession();
    expect(getSession(() => 50_000)).toBeNull();
    stash();
    await completeLogin(CFG, "?code=c&state=st", () => 0);
    expect(getSession(() => 100_001)).toBeNull(); // run out
    expect(sessionStorage.getItem("assay.oidc.session")).toBeNull();
  });

  it("builds a sign-out address only when the provider has a logout endpoint", () => {
    const u = new URL(signOutUrl(CFG)!);
    expect(u.origin + u.pathname).toBe("https://idp.example/logout");
    expect(u.searchParams.get("client_id")).toBe("assay-web");
    expect(signOutUrl({ ...CFG, end_session_endpoint: undefined })).toBeNull();
  });
});
