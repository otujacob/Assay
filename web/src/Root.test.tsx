import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import Root from "./Root";
import { ApiError, api } from "./api";
import * as oidc from "./auth/oidc";

// The App has its own tests. Here it only records how Root started it.
vi.mock("./App", () => ({
  default: (props: { sso?: { user: { key: string; label: string; roles: string[] }; tenant: string; onSignOut: () => void } }) => (
    <div data-testid="app" data-mode={props.sso ? "sso" : "demo"}>
      {props.sso && <><span data-testid="who">{props.sso.user.label}|{props.sso.user.roles.join(",")}|{props.sso.tenant}</span>
        <button type="button" onClick={props.sso.onSignOut}>Sign out</button></>}
    </div>
  ),
}));

const CFG = {
  enabled: true, client_id: "assay-web", authorization_endpoint: "https://idp.example/authorize",
  token_endpoint: "https://idp.example/token", end_session_endpoint: "https://idp.example/logout",
};
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
const me = (roles: string[] = ["analyst"]) => json({ subject: "sso-u1", tenant: "tenant-a", roles, method: "sso" });

function stubServer(routes: Record<string, () => Response>) {
  const calls: { path: string; auth: string | null; demo: string | null }[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: string, init?: RequestInit) => {
    const url = new URL(input, "http://x");
    const h = (init?.headers ?? {}) as Record<string, string>;
    calls.push({ path: url.pathname, auth: h.Authorization ?? null, demo: h["X-Demo-User"] ?? null });
    const r = routes[`${init?.method ?? "GET"} ${url.pathname}`];
    return r ? r() : json({ detail: { code: "not_found" } }, 404);
  }));
  return calls;
}

function session(token = "TOKEN") {
  sessionStorage.setItem("assay.oidc.session", JSON.stringify({ token, expiresAt: Date.now() + 3_600_000, name: "Ada", sub: "u1" }));
}

beforeEach(() => {
  sessionStorage.clear();
  oidc.clearSession();
  window.history.replaceState(null, "", "/");
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("Root", () => {
  it("starts the demo when the server says sign-in is off", async () => {
    stubServer({ "GET /api/auth/config": () => json({ enabled: false }) });
    render(<Root />);
    expect(await screen.findByTestId("app")).toHaveAttribute("data-mode", "demo");
  });

  it("asks a person with no session to sign in, and sends them to the provider", async () => {
    stubServer({ "GET /api/auth/config": () => json(CFG) });
    const to = vi.spyOn(oidc.nav, "to").mockImplementation(() => undefined);
    render(<Root />);
    expect(await screen.findByTestId("sign-in")).toBeInTheDocument();
    expect(screen.queryByTestId("app")).not.toBeInTheDocument(); // nothing of the app before sign-in
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(to).toHaveBeenCalled());
    const u = new URL(to.mock.calls[0][0]);
    expect(u.origin + u.pathname).toBe("https://idp.example/authorize");
    expect(u.searchParams.get("code_challenge_method")).toBe("S256");
    expect(sessionStorage.getItem("assay.oidc.pending")).toContain("verifier"); // kept in this tab to finish the sign-in
  });

  it("finishes a sign-in from the callback, cleans the address bar, and starts the app with the roles the SERVER reports", async () => {
    sessionStorage.setItem("assay.oidc.pending", JSON.stringify({ verifier: "v".repeat(64), state: "st", nonce: "no" }));
    window.history.replaceState(null, "", "/?code=thecode&state=st");
    const id = `h.${btoa(JSON.stringify({ nonce: "no", name: "Ada" }))}.s`;
    const calls = stubServer({
      "GET /api/auth/config": () => json(CFG),
      "POST /token": () => json({ access_token: "TOKEN", id_token: id, expires_in: 600 }),
      "GET /api/me": () => me(["analyst", "auditor"]),
    });
    render(<Root />);
    expect(await screen.findByTestId("who")).toHaveTextContent("Ada|analyst,auditor|tenant-a");
    expect(screen.getByTestId("app")).toHaveAttribute("data-mode", "sso");
    expect(window.location.search).toBe(""); // a reload cannot try the code again
    expect(calls.find((c) => c.path === "/api/me")?.auth).toBe("Bearer TOKEN");
  });

  it("goes back to sign-in with the reason when the provider refuses, or the response is not ours", async () => {
    window.history.replaceState(null, "", "/?error=access_denied&error_description=User+cancelled");
    stubServer({ "GET /api/auth/config": () => json(CFG) });
    const { unmount } = render(<Root />);
    expect(await screen.findByRole("alert")).toHaveTextContent("User cancelled");
    expect(window.location.search).toBe("");
    unmount();
    window.history.replaceState(null, "", "/?code=c&state=forged");
    render(<Root />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/did not match a sign-in started here/);
    expect(oidc.getToken()).toBeNull();
  });

  it("uses an existing session without signing in again, sending it as a bearer token and never the demo header", async () => {
    session();
    const calls = stubServer({ "GET /api/auth/config": () => json(CFG), "GET /api/me": () => me() });
    render(<Root />);
    expect(await screen.findByTestId("who")).toHaveTextContent("Ada|analyst|tenant-a");
    expect(calls.filter((c) => c.path === "/api/me").every((c) => c.auth === "Bearer TOKEN" && c.demo === null)).toBe(true);
  });

  it("tells a person with no role that they have no access, instead of an empty app", async () => {
    session();
    stubServer({ "GET /api/auth/config": () => json(CFG), "GET /api/me": () => me([]) });
    render(<Root />);
    expect(await screen.findByTestId("no-access")).toHaveTextContent(/none of your groups gives you a role/);
    expect(screen.queryByTestId("app")).not.toBeInTheDocument();
  });

  it("asks for MFA again when the API says the sign-in had none, and clears the session", async () => {
    session();
    stubServer({ "GET /api/auth/config": () => json(CFG),
      "GET /api/me": () => json({ detail: { code: "mfa_required", detail: "multi-factor authentication is required" } }, 401) });
    render(<Root />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/did not use multi-factor authentication/);
    expect(oidc.getToken()).toBeNull();
    expect(screen.getByRole("button", { name: "Sign in" })).toBeInTheDocument();
  });

  it("ends the session when the API later stops accepting the token", async () => {
    session();
    let alive = true;
    stubServer({ "GET /api/auth/config": () => json(CFG), "GET /api/me": () => me(),
      "GET /api/dashboard/summary": () => (alive ? json({}) : json({ detail: { code: "unauthorized" } }, 401)) });
    render(<Root />);
    await screen.findByTestId("who");
    alive = false;
    await expect(api.dashboard()).rejects.toBeInstanceOf(ApiError);
    expect(await screen.findByRole("alert")).toHaveTextContent("Your session has ended");
    expect(oidc.getToken()).toBeNull();
    expect(screen.queryByTestId("app")).not.toBeInTheDocument();
  });

  it("signs out: clears the session and goes to the provider's logout when it has one", async () => {
    session();
    stubServer({ "GET /api/auth/config": () => json(CFG), "GET /api/me": () => me() });
    const to = vi.spyOn(oidc.nav, "to").mockImplementation(() => undefined);
    render(<Root />);
    await userEvent.click(await screen.findByRole("button", { name: "Sign out" }));
    expect(oidc.getToken()).toBeNull();
    expect(to.mock.calls[0][0]).toContain("https://idp.example/logout");
  });

  it("signs out to the sign-in screen when the provider has no logout endpoint", async () => {
    session();
    stubServer({ "GET /api/auth/config": () => json({ ...CFG, end_session_endpoint: undefined }), "GET /api/me": () => me() });
    render(<Root />);
    await userEvent.click(await screen.findByRole("button", { name: "Sign out" }));
    expect(await screen.findByTestId("sign-in")).toHaveTextContent("You are signed out.");
    expect(oidc.getToken()).toBeNull();
  });
});
