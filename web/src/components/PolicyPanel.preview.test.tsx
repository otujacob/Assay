import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import type { PolicyPreview, PolicyVersion } from "../types";
import { EMPTY_FORM, PolicyPanel, buildRequest, summarise } from "./PolicyPanel";

const ARROW = "→";
const DOT = "·";
const roles = (...r: string[]) => new Set(r);

const pv = (over: Partial<PolicyVersion> = {}): PolicyVersion => ({
  id: "p1", version: "policy-1", payload: { dq_gate_action: "hold", automation_level: 0 },
  effective_from: "2026-10-01T00:00:00+00:00", proposed_by: "u:admin", status: "pending",
  approved_by: null, approved_at: null, ...over,
});

const preview = (over: Partial<PolicyPreview> = {}): PolicyPreview => ({
  policy: { dq_gate_action: "request_human_review", automation_level: 0, always_review_above: 5000 },
  baseline_version: "policy-0", n_decisions: 10, skipped: 0,
  window: { from: "2026-01-01T00:00:00+00:00", to: "2026-01-01T09:00:00+00:00" },
  review_volume: { before: 5, after: 6, change: 1, change_pct: 20 },
  actions: { before: { approve: 2 }, after: { approve: 1 } }, risk_bands: { before: {}, after: {} },
  changed_decisions: 1, changes: [{ from: "approve", to: "request_human_review", count: 1 }],
  examples: [{ txn_id: "t2", amount: 6000, risk: 0.1, was: "approve", now: "request_human_review", because: "segment_rule" }],
  caveats: ["It does not predict analyst decisions."], ...over,
});

beforeEach(() => vi.restoreAllMocks());

describe("buildRequest", () => {
  const f = (over: Partial<typeof EMPTY_FORM>) => ({ ...EMPTY_FORM, ...over });

  it("leaves blank fields out so the model's own values apply", () => {
    expect(buildRequest(f({}))).toEqual({ dq_gate_action: "request_human_review" });
  });

  it("builds numbers, and a date at the start of the day UTC", () => {
    expect(buildRequest(f({ dq: "hold", tLow: "0.2", tHigh: "0.6", amount: "5000", from: "2026-10-05" }))).toEqual({
      dq_gate_action: "hold", t_low: 0.2, t_high: 0.6, always_review_above: 5000,
      effective_from: "2026-10-05T00:00:00+00:00",
    });
  });

  it.each([
    [{ tLow: "0.2" }, /both risk thresholds/],
    [{ tHigh: "0.6" }, /both risk thresholds/],
    [{ tLow: "0.6", tHigh: "0.2" }, /0 < low < high < 1/],
    [{ tLow: "0.2", tHigh: "1" }, /0 < low < high < 1/],
    [{ tLow: "abc", tHigh: "0.5" }, /0 < low < high < 1/],
    [{ amount: "0" }, /above zero/],
    [{ amount: "-3" }, /above zero/],
    [{ amount: "lots" }, /above zero/],
  ])("explains what is wrong with %j instead of sending it", (over, msg) => {
    expect(buildRequest(f(over))).toMatch(msg);
  });
});

describe("summarise", () => {
  it("lists only the rules a policy actually sets", () => {
    expect(summarise({ dq_gate_action: "request_human_review", automation_level: 0 })).toBe("Review on poor data");
    expect(summarise({ dq_gate_action: "hold", automation_level: 0, t_low: 0.2, t_high: 0.6, always_review_above: 5000 }))
      .toBe(`Risk bands 0.2 / 0.6 ${DOT} Hold on poor data ${DOT} Review amounts from 5,000`);
  });
});

describe("PolicyPanel preview", () => {
  const admin = () => render(<PolicyPanel userKey="admin" roles={roles("admin")} />);

  it("previews the form as typed and shows the change in review volume", async () => {
    vi.spyOn(api, "policies").mockResolvedValue({ items: [] });
    const prev = vi.spyOn(api, "previewPolicy").mockResolvedValue(preview());
    admin();
    await screen.findByText(/No policies yet/);
    fireEvent.change(screen.getByLabelText("Always review amounts from"), { target: { value: "5000" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview impact" }));
    await waitFor(() => expect(prev).toHaveBeenCalledWith({ dq_gate_action: "request_human_review", always_review_above: 5000 }));
    const card = await screen.findByTestId("policy-preview");
    expect(card).toHaveTextContent(`Review volume 5 ${ARROW} 6 (+1, +20%)`);
    expect(card).toHaveTextContent("10 past cases");
    expect(card).toHaveTextContent("compared with policy-0, the policy in force");
    expect(card).toHaveTextContent("It does not predict analyst decisions.");
  });

  it("does not send an invalid form to the server, and says why", async () => {
    vi.spyOn(api, "policies").mockResolvedValue({ items: [] });
    const prev = vi.spyOn(api, "previewPolicy");
    const propose = vi.spyOn(api, "proposePolicy");
    admin();
    await screen.findByText(/No policies yet/);
    fireEvent.change(screen.getByLabelText("Low risk below"), { target: { value: "0.2" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview impact" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Set both risk thresholds, or neither.");
    fireEvent.click(screen.getByRole("button", { name: "Propose policy" }));
    expect(prev).not.toHaveBeenCalled();
    expect(propose).not.toHaveBeenCalled();
  });

  it("clears a preview as soon as the form changes, so it never describes a different policy", async () => {
    vi.spyOn(api, "policies").mockResolvedValue({ items: [] });
    vi.spyOn(api, "previewPolicy").mockResolvedValue(preview());
    admin();
    await screen.findByText(/No policies yet/);
    fireEvent.click(screen.getByRole("button", { name: "Preview impact" }));
    await screen.findByTestId("policy-preview");
    fireEvent.change(screen.getByLabelText("Always review amounts from"), { target: { value: "100" } });
    expect(screen.queryByTestId("policy-preview")).not.toBeInTheDocument();
  });

  it("shows the server's reason when a preview is refused", async () => {
    vi.spyOn(api, "policies").mockResolvedValue({ items: [] });
    vi.spyOn(api, "previewPolicy").mockRejectedValue(new Error("t_low and t_high must be given together"));
    admin();
    await screen.findByText(/No policies yet/);
    fireEvent.click(screen.getByRole("button", { name: "Preview impact" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("given together");
    expect(screen.queryByTestId("policy-preview")).not.toBeInTheDocument();
  });

  it("proposes with the new fields and resets the form", async () => {
    vi.spyOn(api, "policies").mockResolvedValue({ items: [] });
    const propose = vi.spyOn(api, "proposePolicy").mockResolvedValue(pv());
    admin();
    await screen.findByText(/No policies yet/);
    fireEvent.change(screen.getByLabelText("Low risk below"), { target: { value: "0.2" } });
    fireEvent.change(screen.getByLabelText("High risk from"), { target: { value: "0.6" } });
    fireEvent.change(screen.getByLabelText("Always review amounts from"), { target: { value: "5000" } });
    fireEvent.click(screen.getByRole("button", { name: "Propose policy" }));
    await waitFor(() => expect(propose).toHaveBeenCalledWith({
      dq_gate_action: "request_human_review", t_low: 0.2, t_high: 0.6, always_review_above: 5000,
    }));
    await waitFor(() => expect(screen.getByLabelText("Always review amounts from")).toHaveValue(""));
  });

  it("warns that validation was measured at the model's thresholds, only while thresholds are being set", async () => {
    vi.spyOn(api, "policies").mockResolvedValue({ items: [] });
    admin();
    await screen.findByText(/No policies yet/);
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Low risk below"), { target: { value: "0.2" } });
    expect(screen.getByRole("note")).toHaveTextContent("measured at the model’s thresholds");
    fireEvent.change(screen.getByLabelText("Low risk below"), { target: { value: "" } });
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
  });

  it("is not offered to someone who cannot propose", async () => {
    vi.spyOn(api, "policies").mockResolvedValue({ items: [] });
    render(<PolicyPanel userKey="auditor" roles={roles("auditor")} />);
    await screen.findByText(/No policies yet/);
    expect(screen.queryByRole("button", { name: "Preview impact" })).not.toBeInTheDocument();
  });
});
