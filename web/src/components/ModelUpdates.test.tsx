import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import type { Candidate, FeedbackPool, Gate, LearningStatus } from "../types";
import { ModelUpdates, shownGates } from "./ModelUpdates";

const gates = (over: Partial<Record<string, Gate["status"]>> = {}): Gate[] =>
  (["G1", "G2", "G3", "G4", "G6"] as const).map((id) => ({
    id, name: id, status: over[id] ?? (id === "G4" ? "not_evaluated" : id === "G6" ? "review_required" : "pass"), detail: `${id} detail`,
  }));
const cand = (over: Partial<Candidate> = {}): Candidate => ({
  candidate_id: "b-new", base_bundle_id: "b-old", created_by: "u:admin", state: "validated", pool: {},
  training: { extra_labels_used: 120, feedback_used: true },
  gates: { version: "gates-0", gates: gates(), failed: [], unresolved: ["G4", "G6"] }, events: [], ...over,
});
const status: LearningStatus = { champion: "b-old", shadow: null, canary: null, candidates: 1 };
const pool: FeedbackPool = {
  cases: 10, accepted: 4, rejected: 2, deferred: 4, acceptance_rate: 0.5, accepted_by_level: {}, not_accepted_reasons: {},
  flagged: { "an-adv": ["accuracy on matured cases 0.25, below 0.50"] }, note: "unverified labels only count towards a candidate after it passes the gates and is approved",
};
const roles = (...r: string[]) => new Set(r);

function setup(items: Candidate[], st: LearningStatus = status) {
  vi.spyOn(api, "learningStatus").mockResolvedValue(st);
  vi.spyOn(api, "feedbackPool").mockResolvedValue(pool);
  vi.spyOn(api, "candidates").mockResolvedValue({ items });
}

beforeEach(() => vi.restoreAllMocks());

describe("ModelUpdates", () => {
  it("shows the champion, the feedback summary and the quarantined analyst", async () => {
    setup([]);
    render(<ModelUpdates userKey="auditor" roles={roles("auditor")} />);
    expect(await screen.findByTestId("learning-status")).toHaveTextContent("b-old");
    expect(screen.getByTestId("feedback-pool")).toHaveTextContent("4 accepted, 2 rejected, 4 deferred");
    expect(screen.getByRole("note")).toHaveTextContent("an-adv");
    expect(screen.getByText(/No candidate models yet/)).toBeInTheDocument();
  });

  it("shows the shadow and approval gates as met only once the candidate has been approved", () => {
    const withLater = (state: Candidate["state"]) => cand({
      state, gates: { version: "g", gates: [...gates(), { id: "G7", name: "Shadow", status: "pending", detail: "" },
        { id: "G9", name: "Approval", status: "pending", detail: "" }], failed: [], unresolved: [] } });
    const status = (c: Candidate, id: string) => shownGates(c).find((g) => g.id === id)?.status;
    for (const s of ["validated", "shadow", "failed", "rolled_back", "rejected"] as const) {
      expect(status(withLater(s), "G7")).toBe("pending");
    }
    for (const s of ["approved", "canary", "champion"] as const) {
      expect(status(withLater(s), "G7")).toBe("pass");
      expect(status(withLater(s), "G9")).toBe("pass");
      expect(status(withLater(s), "G4")).toBe("not_evaluated"); // a gate that could not be judged stays that way
    }
  });

  it("refreshes on request, so a running shadow can be watched", async () => {
    setup([cand({ state: "shadow", shadow: { status: "pending", detail: "3 cases over 0.0 days", n: 3, alarms: [] } })]);
    render(<ModelUpdates userKey="auditor" roles={roles("auditor")} />);
    expect(await screen.findByTestId("shadow-report")).toHaveTextContent("3 cases");
    vi.spyOn(api, "candidates").mockResolvedValue({ items: [cand({ state: "shadow", shadow: { status: "pass", detail: "60 cases over 2.0 days with no alarm", n: 60, alarms: [] } })] });
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() => expect(screen.getByTestId("shadow-report")).toHaveTextContent("60 cases"));
  });

  it("lists every gate with a word for its state, and never calls an unjudged gate a pass", async () => {
    setup([cand()]);
    render(<ModelUpdates userKey="auditor" roles={roles("auditor")} />);
    const table = await screen.findByRole("table", { name: "Gates for b-new" });
    expect(table).toHaveTextContent("Could not judge");
    expect(table).toHaveTextContent("Needs review");
    expect(table).toHaveTextContent("Pass");
    expect(screen.queryByRole("button", { name: "Start shadow" })).not.toBeInTheDocument(); // an auditor reads only
    expect(screen.queryByRole("button", { name: "Reject" })).not.toBeInTheDocument();
  });

  it("an administrator can start shadow, and the step is sent to the server", async () => {
    setup([cand()]);
    const step = vi.spyOn(api, "candidateStep").mockResolvedValue(cand({ state: "shadow" }));
    render(<ModelUpdates userKey="admin" roles={roles("admin")} />);
    fireEvent.click(await screen.findByRole("button", { name: "Start shadow" }));
    await waitFor(() => expect(step).toHaveBeenCalledWith("b-new", "shadow", {}));
  });

  it("a candidate that failed a gate offers no way forward", async () => {
    setup([cand({ state: "failed", gates: { version: "g", gates: gates({ G2: "fail" }), failed: ["G2"], unresolved: [] } })]);
    render(<ModelUpdates userKey="approver" roles={roles("approver")} />);
    expect(await screen.findByText("Failed a gate")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start shadow" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reject" })).toBeDisabled(); // needs a reason first
  });

  it("approval needs a rationale, the segment confirmation, and a waiver for each gate that could not be judged", async () => {
    setup([cand({ state: "shadow", shadow: { status: "pass", detail: "300 cases over 8.0 days with no alarm", n: 300, agreement: 0.97, alarms: [] } })]);
    const step = vi.spyOn(api, "candidateStep").mockResolvedValue(cand({ state: "approved" }));
    render(<ModelUpdates userKey="approver" roles={roles("approver")} />);
    expect(await screen.findByTestId("shadow-report")).toHaveTextContent("Agrees with the champion on 97%");
    const approve = screen.getByRole("button", { name: "Approve" });
    expect(approve).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Approval rationale"), { target: { value: "reviewed" } });
    expect(approve).toBeDisabled(); // still needs the segment confirmation
    fireEvent.click(screen.getByLabelText(/I reviewed the results by segment/));
    expect(approve).toBeEnabled();
    fireEvent.click(screen.getByLabelText("G4")); // waive the gate that could not be judged
    fireEvent.click(approve);
    await waitFor(() => expect(step).toHaveBeenCalledWith("b-new", "approve", {
      rationale: "reviewed", reviewed_segments: true, waived: ["G4"] }));
  });

  it("the person who created a candidate cannot approve it", async () => {
    setup([cand({ state: "shadow", created_by: "u:approver" })]);
    render(<ModelUpdates userKey="approver" roles={roles("approver")} />);
    expect(await screen.findByText(/someone else must approve it/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Approval rationale"), { target: { value: "mine" } });
    fireEvent.click(screen.getByLabelText(/I reviewed the results by segment/));
    expect(screen.getByRole("button", { name: "Approve" })).toBeDisabled();
  });

  it("an alarm in the shadow report is shown", async () => {
    setup([cand({ state: "shadow", shadow: { status: "fail", detail: "x", n: 300, alarms: ["flag rate differs from the champion's by 0.080"] } })]);
    render(<ModelUpdates userKey="auditor" roles={roles("auditor")} />);
    expect(await screen.findByTestId("shadow-report")).toHaveTextContent("flag rate differs");
  });

  it("canary needs a share, promotion is one click, and rollback needs a reason", async () => {
    setup([cand({ state: "approved" })]);
    const step = vi.spyOn(api, "candidateStep").mockResolvedValue(cand({ state: "canary" }));
    const { unmount } = render(<ModelUpdates userKey="approver" roles={roles("approver")} />);
    fireEvent.change(await screen.findByLabelText("Canary share"), { target: { value: "0.2" } });
    fireEvent.click(screen.getByRole("button", { name: "Start canary" }));
    await waitFor(() => expect(step).toHaveBeenCalledWith("b-new", "canary", { share: 0.2 }));
    unmount();

    setup([cand({ state: "canary" })]);
    render(<ModelUpdates userKey="approver" roles={roles("approver")} />);
    fireEvent.click(await screen.findByRole("button", { name: "Promote to champion" }));
    await waitFor(() => expect(step).toHaveBeenCalledWith("b-new", "promote", {}));
    const rollback = screen.getByRole("button", { name: "Roll back" });
    expect(rollback).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Roll back (reason)"), { target: { value: "calibration breach" } });
    fireEvent.click(rollback);
    await waitFor(() => expect(step).toHaveBeenCalledWith("b-new", "rollback", { reason: "calibration breach" }));
  });

  it("shows the server's refusal instead of hiding it", async () => {
    setup([cand()]);
    vi.spyOn(api, "candidateStep").mockRejectedValue(new Error("shadow evidence is not sufficient: 0 cases"));
    render(<ModelUpdates userKey="admin" roles={roles("admin")} />);
    fireEvent.click(await screen.findByRole("button", { name: "Start shadow" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("shadow evidence is not sufficient");
  });

  it("says so when the data cannot be loaded", async () => {
    vi.spyOn(api, "learningStatus").mockRejectedValue(new Error("forbidden"));
    vi.spyOn(api, "feedbackPool").mockRejectedValue(new Error("forbidden"));
    vi.spyOn(api, "candidates").mockRejectedValue(new Error("forbidden"));
    render(<ModelUpdates userKey="admin" roles={roles("admin")} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load model updates");
  });
});
