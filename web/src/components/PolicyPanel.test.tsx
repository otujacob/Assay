import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import type { PolicyVersion } from "../types";
import { PolicyPanel, inForce } from "./PolicyPanel";

const pv = (over: Partial<PolicyVersion> = {}): PolicyVersion => ({
  id: "p1", version: "policy-1", payload: { dq_gate_action: "hold", automation_level: 0 },
  effective_from: "2026-10-01T00:00:00+00:00", proposed_by: "u:admin", status: "pending",
  approved_by: null, approved_at: null, ...over,
});
const roles = (...r: string[]) => new Set(r);

beforeEach(() => vi.restoreAllMocks());

describe("PolicyPanel", () => {
  it("lists policies with who proposed them and whether they await approval", async () => {
    vi.spyOn(api, "policies").mockResolvedValue({ items: [pv()] });
    render(<PolicyPanel userKey="auditor" roles={roles("auditor")} />);
    expect(await screen.findByText("policy-1")).toBeInTheDocument();
    expect(screen.getByText("Hold on poor data")).toBeInTheDocument();
    expect(screen.getByText("Awaiting approval")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument(); // read-only roles cannot act
    expect(screen.queryByRole("button", { name: "Propose policy" })).not.toBeInTheDocument();
  });

  it("marks the approved policy as in force, and says nothing is in force when there is none", async () => {
    const spy = vi.spyOn(api, "policies").mockResolvedValue({
      items: [pv({ status: "approved", approved_by: "u:approver", approved_at: "2026-10-02T00:00:00+00:00" })],
    });
    const { unmount } = render(<PolicyPanel userKey="auditor" roles={roles("auditor")} />);
    expect(await screen.findByText("In force")).toBeInTheDocument();
    expect(screen.getByText("Approved by u:approver")).toBeInTheDocument();
    unmount();
    spy.mockResolvedValue({ items: [] });
    render(<PolicyPanel userKey="auditor" roles={roles("auditor")} />);
    expect(await screen.findByText(/No policies yet/)).toBeInTheDocument();
  });

  it("lets an administrator propose, sending the chosen gate action and date", async () => {
    vi.spyOn(api, "policies").mockResolvedValue({ items: [] });
    const propose = vi.spyOn(api, "proposePolicy").mockResolvedValue(pv());
    render(<PolicyPanel userKey="admin" roles={roles("admin")} />);
    await screen.findByText(/No policies yet/);
    fireEvent.change(screen.getByLabelText("Data-quality gate action"), { target: { value: "hold" } });
    fireEvent.change(screen.getByLabelText("Effective from"), { target: { value: "2026-10-05" } });
    fireEvent.click(screen.getByRole("button", { name: "Propose policy" }));
    await waitFor(() => expect(propose).toHaveBeenCalledWith({
      dq_gate_action: "hold", effective_from: "2026-10-05T00:00:00+00:00",
    }));
  });

  it("lets an approver approve someone else's policy", async () => {
    vi.spyOn(api, "policies").mockResolvedValue({ items: [pv({ proposed_by: "u:admin" })] });
    const approve = vi.spyOn(api, "approvePolicy").mockResolvedValue(pv({ status: "approved" }));
    render(<PolicyPanel userKey="approver" roles={roles("approver")} />);
    const row = (await screen.findByText("policy-1")).closest("tr")!;
    fireEvent.click(within(row).getByRole("button", { name: "Approve" }));
    await waitFor(() => expect(approve).toHaveBeenCalledWith("p1"));
  });

  it("disables approval on a policy the user proposed, and explains why", async () => {
    vi.spyOn(api, "policies").mockResolvedValue({ items: [pv({ proposed_by: "u:both" })] });
    render(<PolicyPanel userKey="both" roles={roles("admin", "approver")} />);
    const btn = await screen.findByRole("button", { name: "Approve" });
    expect(btn).toBeDisabled();
    expect(btn).toHaveAttribute("title", expect.stringMatching(/someone else must approve/));
  });

  it("shows the server's reason when it refuses, and does not pretend it worked", async () => {
    vi.spyOn(api, "policies").mockResolvedValue({ items: [pv()] });
    vi.spyOn(api, "approvePolicy").mockRejectedValue(new Error("this policy version is already approved"));
    render(<PolicyPanel userKey="approver" roles={roles("approver")} />);
    fireEvent.click(await screen.findByRole("button", { name: "Approve" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("already approved");
  });

  it("reports a load failure instead of an empty table", async () => {
    vi.spyOn(api, "policies").mockRejectedValue(new Error("forbidden"));
    render(<PolicyPanel userKey="analyst" roles={roles("admin")} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load policies: forbidden");
    expect(screen.queryByText(/No policies yet/)).not.toBeInTheDocument();
  });
});

describe("inForce", () => {
  const approved = (version: string, effective: string, approvedAt: string): PolicyVersion =>
    pv({ id: version, version, status: "approved", effective_from: effective, approved_at: approvedAt, approved_by: "u:x" });
  const NOW = Date.parse("2026-10-10T00:00:00Z");

  it("ignores pending policies and policies that take effect in the future", () => {
    const items = [approved("policy-3", "2026-10-20T00:00:00Z", "2026-10-05T00:00:00Z"), pv({ version: "policy-2" }),
                   approved("policy-1", "2026-10-01T00:00:00Z", "2026-10-02T00:00:00Z")];
    expect(inForce(items, NOW)?.version).toBe("policy-1");   // policy-3 is approved but scheduled
    expect(inForce([items[1]], NOW)).toBeUndefined();        // nothing approved: the built-in default applies
  });

  it("uses the later of the effective date and the approval time", () => {
    const backdated = approved("policy-2", "2026-09-01T00:00:00Z", "2026-10-08T00:00:00Z"); // took effect on approval
    const earlier = approved("policy-1", "2026-10-01T00:00:00Z", "2026-10-02T00:00:00Z");
    expect(inForce([backdated, earlier], NOW)?.version).toBe("policy-2");
  });

  it("lets the newest policy win when two took effect at the same moment", () => {
    const a = approved("policy-1", "2026-10-01T00:00:00Z", "2026-10-01T00:00:00Z");
    const b = approved("policy-2", "2026-10-01T00:00:00Z", "2026-10-01T00:00:00Z");
    expect(inForce([b, a], NOW)?.version).toBe("policy-2");  // the list is newest first
  });

  it("shows an approved but not yet effective policy as Scheduled, not In force", async () => {
    const future = new Date(Date.now() + 7 * 864e5).toISOString();
    vi.spyOn(api, "policies").mockResolvedValue({ items: [approved("policy-1", future, new Date().toISOString())] });
    render(<PolicyPanel userKey="auditor" roles={roles("auditor")} />);
    expect(await screen.findByText("Scheduled")).toBeInTheDocument();
    expect(screen.queryByText("In force")).not.toBeInTheDocument();
  });
});
