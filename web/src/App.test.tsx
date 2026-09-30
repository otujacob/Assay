import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { setUser } from "./api";
import { bundle, caseView, dashboard, item, report } from "./test/fixtures";

type Handler = (url: URL, init?: RequestInit) => { status?: number; body: unknown };

const USERS = [
  { key: "analyst", label: "Analyst", roles: ["analyst"] },
  { key: "manager", label: "Fraud Manager", roles: ["manager"] },
  { key: "auditor", label: "Auditor", roles: ["auditor"] },
];

function mockApi(handlers: Record<string, Handler>) {
  const calls: { path: string; method: string; user: string | null; body: unknown }[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: string, init?: RequestInit) => {
    const url = new URL(input, "http://x");
    const headers = (init?.headers ?? {}) as Record<string, string>;
    calls.push({ path: url.pathname + url.search, method: init?.method ?? "GET", user: headers["X-Demo-User"] ?? null,
      body: init?.body ? JSON.parse(init.body as string) : null });
    const key = `${init?.method ?? "GET"} ${url.pathname}`;
    const h = handlers[key];
    const out = h ? h(url, init) : { status: 404, body: { detail: { code: "not_found" } } };
    return new Response(JSON.stringify(out.body), { status: out.status ?? 200 });
  }));
  return calls;
}

const queue = (items = [item(), item({ decision_id: "d-2", txn_id: "t-0002", priority: "P2" })]) =>
  ({ body: { items, matching: items.length, total_cases: 9, sla_breached: 0 } });

beforeEach(() => {
  localStorage.clear();
  setUser("analyst"); // the client keeps the demo user in module state; reset it between tests
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("loads the queue and opens the highest-priority case", async () => {
    mockApi({
      "GET /demo/users": () => ({ body: USERS }),
      "GET /api/review/queue": () => queue(),
      "GET /api/review/cases/d-1": () => ({ body: caseView() }),
      "GET /api/dashboard/summary": () => ({ status: 403, body: { detail: { code: "forbidden" } } }),
      "GET /api/validation/reports": () => ({ status: 403, body: { detail: { code: "forbidden" } } }),
      "GET /api/models/bundles": () => ({ status: 403, body: { detail: { code: "forbidden" } } }),
    });
    render(<App />);
    expect(await screen.findByTestId("case-details")).toHaveTextContent("t-0001");
    expect(screen.getByTestId("row-t-0002")).toBeInTheDocument();
    expect(screen.getByText(/DEMO: synthetic data/)).toBeInTheDocument();
    expect(screen.getByText(/Recommend-Only Mode/)).toBeInTheDocument();
  });

  it("handles a role the API refuses without showing an error banner or fake numbers", async () => {
    mockApi({
      "GET /demo/users": () => ({ body: USERS }),
      "GET /api/review/queue": () => queue(),
      "GET /api/review/cases/d-1": () => ({ body: caseView() }),
    });
    render(<App />);
    await screen.findByTestId("case-details");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByText(/Dashboard figures are available to the manager role/)).toBeInTheDocument();
    expect(screen.getByText(/Governance evidence .* is available to the auditor/)).toBeInTheDocument();
  });

  it("reloads as a different demo user and sends that user with every request", async () => {
    const calls = mockApi({
      "GET /demo/users": () => ({ body: USERS }),
      "GET /api/review/queue": () => queue(),
      "GET /api/review/cases/d-1": () => ({ body: caseView() }),
      "GET /api/dashboard/summary": () => ({ body: dashboard() }),
      "GET /api/validation/reports": () => ({ body: [report()] }),
      "GET /api/models/bundles": () => ({ body: [bundle()] }),
    });
    render(<App />);
    await screen.findByTestId("case-details");
    fireEvent.change(screen.getByLabelText("Demo user"), { target: { value: "manager" } });
    expect(await screen.findByTestId("kpis")).toHaveTextContent("Total Pending Reviews");
    expect(await screen.findByTestId("bundle-card").catch(() => null)).toBeNull(); // bundle tab is not the default
    fireEvent.click(await screen.findByRole("tab", { name: "Model Bundle & Lineage" }));
    expect(screen.getByTestId("bundle-card")).toHaveTextContent("b-1");
    expect(calls.some((c) => c.user === "manager" && c.path.startsWith("/api/dashboard"))).toBe(true);
    expect(localStorage.getItem("assay.demoUser")).toBe("manager");
  });

  it("switches case when a row is clicked", async () => {
    mockApi({
      "GET /demo/users": () => ({ body: USERS }),
      "GET /api/review/queue": () => queue(),
      "GET /api/review/cases/d-1": () => ({ body: caseView() }),
      "GET /api/review/cases/d-2": () => ({ body: caseView({ decision_id: "d-2", txn_id: "t-0002" }) }),
    });
    render(<App />);
    await screen.findByTestId("case-details");
    fireEvent.click(screen.getByTestId("row-t-0002"));
    await waitFor(() => expect(screen.getByTestId("case-details")).toHaveTextContent("t-0002"));
  });

  it("records a decision, then refreshes the queue and the case", async () => {
    let decided = false;
    const calls = mockApi({
      "GET /demo/users": () => ({ body: USERS }),
      "GET /api/review/queue": () => queue(decided ? [item({ decision_id: "d-2", txn_id: "t-0002" })] : undefined),
      "GET /api/review/cases/d-1": () => ({ body: caseView(decided ? { status: "decided", final_decision: "approve" } : {}) }),
      "GET /api/review/cases/d-2": () => ({ body: caseView({ decision_id: "d-2", txn_id: "t-0002" }) }),
      "POST /api/review/cases/d-1/actions": () => {
        decided = true;
        return { status: 201, body: { action_id: "a", status: "decided", final_decision: "approve", needs_second_review: false,
          feedback: { aas: null, fcs: 0.5, lvs: 0.2, crs: null, fqs: 0.4, disposition: "defer", reason: "awaiting outcome", formula_version: "fqs-0", note: "s" } } };
      },
    });
    render(<App />);
    const panel = await screen.findByTestId("action-panel");
    fireEvent.change(within(panel).getByLabelText("Reason code"), { target: { value: "customer_denied" } });
    fireEvent.click(within(panel).getByRole("button", { name: /Block/ }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    const post = calls.find((c) => c.method === "POST")!;
    expect(post.path).toBe("/api/review/cases/d-1/actions");
    expect(post.body).toMatchObject({ action: "block", reason_code: "customer_denied" });
    await waitFor(() => expect(screen.queryByTestId("row-t-0001")).not.toBeInTheDocument()); // left the queue
  });

  it("shows a server validation error instead of pretending the action worked", async () => {
    mockApi({
      "GET /demo/users": () => ({ body: USERS }),
      "GET /api/review/queue": () => queue(),
      "GET /api/review/cases/d-1": () => ({ body: caseView() }),
      "POST /api/review/cases/d-1/actions": () => ({ status: 409, body: { detail: { code: "already_decided", detail: "this analyst has already decided this case" } } }),
    });
    render(<App />);
    const panel = await screen.findByTestId("action-panel");
    fireEvent.click(within(panel).getByRole("button", { name: /Escalate/ }));
    expect(await within(panel).findByRole("alert")).toHaveTextContent("this analyst has already decided this case");
    expect(screen.queryByTestId("action-result")).not.toBeInTheDocument();
  });

  it("tells the user when the server cannot be reached", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("nope", { status: 502 })));
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/Cannot reach the demo server/);
  });

  it("applies queue filters on the server, where redaction is known", async () => {
    const calls = mockApi({
      "GET /demo/users": () => ({ body: USERS }),
      "GET /api/review/queue": () => queue(),
      "GET /api/review/cases/d-1": () => ({ body: caseView() }),
    });
    render(<App />);
    await screen.findByTestId("case-details");
    fireEvent.change(screen.getByLabelText("Filter by trust band"), { target: { value: "low" } });
    await waitFor(() => expect(calls.some((c) => c.path.includes("trust_state=low"))).toBe(true));
    fireEvent.change(screen.getByLabelText("Filter by risk band"), { target: { value: "high" } });
    await waitFor(() => expect(calls.some((c) => c.path.includes("risk_band=high") && c.path.includes("trust_state=low"))).toBe(true));
  });

  it("returns to an empty state from Back to Queue", async () => {
    mockApi({
      "GET /demo/users": () => ({ body: USERS }),
      "GET /api/review/queue": () => queue(),
      "GET /api/review/cases/d-1": () => ({ body: caseView() }),
    });
    render(<App />);
    await screen.findByTestId("case-details");
    fireEvent.click(screen.getByRole("button", { name: /Back to Queue/ }));
    expect(await screen.findByText(/Select a case from the queue/)).toBeInTheDocument();
  });
});
