import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import type { AuditResponse } from "../types";
import { AuditLog, toFilters } from "./AuditLog";

const row = (n: number, action = "score") => ({
  seq: n, time: "2026-09-30T12:00:00+00:00", actor: "api:ing", action, object: `t-${n}`, result: "ok", row_hash: `h${n}`,
});
const resp = (over: Partial<AuditResponse> = {}): AuditResponse => ({
  items: [row(2), row(1, "ingest_transaction")], matching: 2, chain_ok: true, ...over,
});

beforeEach(() => vi.restoreAllMocks());

describe("toFilters", () => {
  it("drops blanks and makes the To date inclusive by using the next day as the exclusive bound", () => {
    expect(toFilters({ txn: " t-1 ", actor: "", action: "", from: "2026-09-30", to: "2026-09-30" })).toEqual({
      txn_id: "t-1", actor: undefined, action: undefined, since: "2026-09-30T00:00:00", until: "2026-10-01T00:00:00",
    });
  });
});

describe("AuditLog", () => {
  it("loads on mount and shows rows, the count and the verified hash chain", async () => {
    vi.spyOn(api, "audit").mockResolvedValue(resp());
    render(<AuditLog />);
    expect(await screen.findByText("t-2")).toBeInTheDocument();
    expect(screen.getByText("ingest_transaction")).toBeInTheDocument();
    expect(screen.getByText(/Showing 2 of 2/)).toBeInTheDocument();
    expect(screen.getByText("Hash chain verified")).toBeInTheDocument();
  });

  it("flags a broken hash chain loudly", async () => {
    vi.spyOn(api, "audit").mockResolvedValue(resp({ chain_ok: false }));
    render(<AuditLog />);
    expect(await screen.findByText("Hash chain BROKEN")).toBeInTheDocument();
  });

  it("searches with the filters the auditor typed", async () => {
    const spy = vi.spyOn(api, "audit").mockResolvedValue(resp());
    render(<AuditLog />);
    await screen.findByText("t-2");
    fireEvent.change(screen.getByLabelText("Transaction"), { target: { value: "t-1" } });
    fireEvent.change(screen.getByLabelText("Action"), { target: { value: "score" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(spy).toHaveBeenLastCalledWith(expect.objectContaining({ txn_id: "t-1", action: "score" })));
  });

  it("shows an empty state, and an error when the server refuses", async () => {
    vi.spyOn(api, "audit").mockResolvedValueOnce(resp({ items: [], matching: 0 }));
    render(<AuditLog />);
    expect(await screen.findByText("No records match.")).toBeInTheDocument();
    vi.spyOn(api, "audit").mockRejectedValueOnce(new Error("forbidden"));
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("forbidden");
  });

  it("exports the same filters as CSV", async () => {
    vi.spyOn(api, "audit").mockResolvedValue(resp());
    const csv = vi.spyOn(api, "auditCsv").mockResolvedValue(new Blob(["x"]));
    Object.assign(URL, { createObjectURL: vi.fn(() => "blob:x"), revokeObjectURL: vi.fn() });
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {}); // jsdom cannot navigate
    render(<AuditLog />);
    await screen.findByText("t-2");
    fireEvent.change(screen.getByLabelText("Actor"), { target: { value: "api:ing" } });
    fireEvent.click(screen.getByRole("button", { name: "Export CSV" }));
    await waitFor(() => expect(csv).toHaveBeenCalledWith(expect.objectContaining({ actor: "api:ing" })));
  });
});
