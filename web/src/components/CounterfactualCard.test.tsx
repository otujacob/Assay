import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import type { CounterfactualItem, CounterfactualView } from "../types";
import { CounterfactualCard } from "./CounterfactualCard";

const ARROW = "→";

const item = (over: Partial<CounterfactualItem> = {}): CounterfactualItem => ({
  groups: ["counterparty"], changes: [{ feature: "new_beneficiary", from: 1, to: 0, text: `a new beneficiary: yes ${ARROW} no` }],
  risk_before: 0.95, risk_after: 0.02, distance: 3.1, valid: true, robust_share: 1, failed_checks: [], ...over,
});

const view = (over: Partial<CounterfactualView> = {}): CounterfactualView => ({
  decision_id: "d-1", risk: 0.95, threshold: 0.12, flagged: true, counterfactuals: [item()],
  summary: { found: 1, valid: 1, score: 1 },
  note: "A counterfactual describes the MODEL, not cause and effect. It is not advice and must not be given to a customer (PRD 7.4).",
  cross_method: { agreement: 0.9, shap_drivers: ["amount", "counterparty"], permutation_drivers: ["amount", "counterparty"] },
  ...over,
});

beforeEach(() => vi.restoreAllMocks());

const open = async (v: CounterfactualView) => {
  vi.spyOn(api, "counterfactuals").mockResolvedValue(v);
  render(<CounterfactualCard decisionId="d-1" />);
  fireEvent.click(screen.getByRole("button", { name: "Show what-ifs" }));
  await screen.findByTestId("cf-note");
};

describe("CounterfactualCard", () => {
  it("does nothing until asked, because working it out re-scores hundreds of variations", () => {
    const spy = vi.spyOn(api, "counterfactuals");
    render(<CounterfactualCard decisionId="d-1" />);
    expect(spy).not.toHaveBeenCalled();
    expect(screen.getByText(/not what caused anything/)).toBeInTheDocument();
  });

  it("says what would flip a flagged call, in words, with the risk before and after", async () => {
    await open(view());
    const li = screen.getByTestId("cf-list");
    expect(li).toHaveTextContent(`Would not be flagged if a new beneficiary: yes ${ARROW} no`);
    expect(li).toHaveTextContent(`risk 95.0% ${ARROW} 2.0%`);
    expect(li).toHaveTextContent("changes beneficiary");
    expect(screen.getByText(/flagged this at 95.0% \(threshold 12.0%\)/)).toBeInTheDocument();
  });

  it("explains the other direction for a case that was not flagged", async () => {
    await open(view({ flagged: false, risk: 0.03, counterfactuals: [item({ risk_before: 0.03, risk_after: 0.4 })] }));
    expect(screen.getByTestId("cf-list")).toHaveTextContent("Would be flagged if");
    expect(screen.getByText(/did not flag this at 3.0%/)).toBeInTheDocument();
  });

  it("always shows the caveat that this describes the model and is not advice", async () => {
    await open(view());
    expect(screen.getByTestId("cf-note")).toHaveTextContent("describes the MODEL, not cause and effect");
    expect(screen.getByTestId("cf-note")).toHaveTextContent("must not be given to a customer");
  });

  it("lists only counterfactuals that passed every check and held up, never the doubtful ones", async () => {
    await open(view({ counterfactuals: [
      item(),
      item({ valid: false, failed_checks: ["robust"], changes: [{ feature: "hour", from: 3, to: 4, text: "hour of day 03:00 → 04:00" }] }),
      item({ robust_share: 0.5, changes: [{ feature: "amount", from: 5, to: 9, text: "amount 5 → 9" }] }),
    ] }));
    const list = screen.getByTestId("cf-list");
    expect(list.querySelectorAll("li")).toHaveLength(1);
    expect(list).not.toHaveTextContent("hour of day");
    expect(list).not.toHaveTextContent("amount 5");
  });

  it("says plainly when candidates were found but none held up", async () => {
    await open(view({ counterfactuals: [item({ valid: false, failed_checks: ["robust"] })] }));
    expect(screen.queryByTestId("cf-list")).not.toBeInTheDocument();
    expect(screen.getByText(/none held up under the checks/)).toBeInTheDocument();
  });

  it("says plainly when no realistic change exists, as information and not as a failure", async () => {
    await open(view({ counterfactuals: [], summary: { found: 0, score: null } }));
    expect(screen.getByTestId("cf-none")).toHaveTextContent("It is firm");
    expect(screen.queryByTestId("cf-list")).not.toBeInTheDocument();
  });

  it("shows how far the two explanation methods agree, and warns when they do not", async () => {
    await open(view());
    expect(screen.getByTestId("cf-methods")).toHaveTextContent("Two methods agree 90%");
    expect(screen.getByTestId("cf-methods")).toHaveTextContent("SHAP names amount, beneficiary");
    expect(screen.getByTestId("cf-methods")).not.toHaveTextContent("extra care");
    vi.restoreAllMocks();
    vi.spyOn(api, "counterfactuals").mockResolvedValue(view({
      cross_method: { agreement: 0.3, shap_drivers: ["amount"], permutation_drivers: ["velocity", "timing"] },
    }));
    const { unmount } = render(<CounterfactualCard decisionId="d-2" />);
    fireEvent.click(screen.getAllByRole("button", { name: "Show what-ifs" })[0]);
    await waitFor(() => expect(screen.getAllByTestId("cf-methods").some((n) => /extra care/.test(n.textContent ?? ""))).toBe(true));
    unmount();
  });

  it("shows the server's reason when it cannot answer, rather than an empty card", async () => {
    vi.spyOn(api, "counterfactuals").mockRejectedValue(new Error("this case is under blind review until you record a decision"));
    render(<CounterfactualCard decisionId="d-1" />);
    fireEvent.click(screen.getByRole("button", { name: "Show what-ifs" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("blind review");
    expect(screen.queryByTestId("cf-note")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show what-ifs" })).toBeEnabled();      // and the user can try again
  });

  it("disables the button while it works", async () => {
    let release!: (v: CounterfactualView) => void;
    vi.spyOn(api, "counterfactuals").mockReturnValue(new Promise<CounterfactualView>((r) => { release = r; }));
    render(<CounterfactualCard decisionId="d-1" />);
    fireEvent.click(screen.getByRole("button", { name: "Show what-ifs" }));
    expect(await screen.findByRole("button", { name: /Working it out/ })).toBeDisabled();
    release(view());
    await screen.findByTestId("cf-note");
  });
});
