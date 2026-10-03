import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { PolicyPreview } from "../types";
import { PolicyPreviewCard, volumeSummary } from "./PolicyPreviewCard";

const ARROW = "→";

const base = (over: Partial<PolicyPreview> = {}): PolicyPreview => ({
  policy: { dq_gate_action: "request_human_review", automation_level: 0 }, baseline_version: "policy-3", n_decisions: 40,
  skipped: 0, window: { from: "2026-01-01T00:00:00+00:00", to: "2026-01-03T00:00:00+00:00" },
  review_volume: { before: 10, after: 7, change: -3, change_pct: -30 }, actions: { before: {}, after: {} },
  risk_bands: { before: {}, after: {} }, changed_decisions: 3,
  changes: [{ from: "request_human_review", to: "approve", count: 3 }], examples: [], caveats: ["A caveat."], ...over,
});

describe("volumeSummary", () => {
  it.each([
    [{ before: 5, after: 8, change: 3, change_pct: 60 }, `Review volume 5 ${ARROW} 8 (+3, +60%)`],
    [{ before: 10, after: 7, change: -3, change_pct: -30 }, `Review volume 10 ${ARROW} 7 (-3, -30%)`],
    [{ before: 5, after: 5, change: 0, change_pct: 0 }, "No change in review volume (5 cases, 0%)"],
    [{ before: 0, after: 4, change: 4, change_pct: null }, `Review volume 0 ${ARROW} 4 (+4)`], // no percentage from a zero base
    [{ before: 0, after: 0, change: 0, change_pct: null }, "No change in review volume (0 cases)"],
  ])("says %j in plain words", (v, text) => expect(volumeSummary(v)).toBe(text));
});

describe("PolicyPreviewCard", () => {
  it("lists what would change, in words, with the caveats", () => {
    render(<PolicyPreviewCard preview={base()} />);
    expect(screen.getByText("Human review")).toBeInTheDocument();
    expect(screen.getByText("Approve")).toBeInTheDocument();
    expect(screen.getByTestId("policy-preview")).toHaveTextContent(`Review volume 10 ${ARROW} 7 (-3, -30%)`);
    expect(screen.getByText("A caveat.")).toBeInTheDocument();
  });

  it("says plainly when nothing would change", () => {
    render(<PolicyPreviewCard preview={base({
      changed_decisions: 0, changes: [], review_volume: { before: 4, after: 4, change: 0, change_pct: 0 },
    })} />);
    expect(screen.getByText("No decision would have changed.")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("does not pretend there is a result when there is no history", () => {
    render(<PolicyPreviewCard preview={base({
      n_decisions: 0, changed_decisions: 0, changes: [], window: { from: null, to: null },
      review_volume: { before: 0, after: 0, change: 0, change_pct: null },
    })} />);
    expect(screen.getByText("There are no past decisions to replay yet.")).toBeInTheDocument();
    expect(screen.queryByText("No decision would have changed.")).not.toBeInTheDocument();
  });

  it("reports cases that could not be replayed", () => {
    render(<PolicyPreviewCard preview={base({ skipped: 6 })} />);
    expect(screen.getByTestId("policy-preview")).toHaveTextContent("6 could not be replayed.");
  });

  it("explains why an example changed in the institution's terms", () => {
    render(<PolicyPreviewCard preview={base({ examples: [{
      txn_id: "t2", amount: 6000, risk: 0.1, was: "approve", now: "request_human_review", because: "segment_rule",
    }] })} />);
    expect(screen.getByText("amount rule")).toBeInTheDocument();
    expect(screen.getByText("6,000")).toBeInTheDocument();
  });
});
