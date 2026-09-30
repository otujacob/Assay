import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { item } from "../test/fixtures";
import { EMPTY_FILTERS, ReviewQueue } from "./ReviewQueue";

const base = { selectedId: null, onSelect: () => {}, filters: EMPTY_FILTERS, onFilters: () => {}, elapsedSeconds: 0, total: 3, loading: false };

describe("ReviewQueue", () => {
  it("renders one row per case with priority, risk, trust, amount and SLA", () => {
    render(<ReviewQueue {...base} items={[item(), item({ decision_id: "d-2", txn_id: "t-0002", priority: "P3" })]} />);
    const row = screen.getByTestId("row-t-0001");
    expect(within(row).getByText("P1")).toBeInTheDocument();
    expect(within(row).getByText("High")).toBeInTheDocument();
    expect(within(row).getByText("Moderate")).toBeInTheDocument();
    expect(within(row).getByText("01:18:32")).toBeInTheDocument();
    expect(within(row).getByText(/1,250\.50/)).toBeInTheDocument();
    expect(screen.getByTestId("row-t-0002")).toBeInTheDocument();
  });

  it("hides risk and trust on a blind case and marks it", () => {
    render(<ReviewQueue {...base} items={[item({ blind: true, redacted: true, risk_band: null, trust_state: null })]} />);
    const row = screen.getByTestId("row-t-0001");
    expect(within(row).getAllByText("hidden")).toHaveLength(2);
    expect(within(row).getByTitle(/Blind review/)).toBeInTheDocument();
    expect(within(row).queryByText("Moderate")).not.toBeInTheDocument();
  });

  it("shows Insufficient Evidence as a label rather than a number", () => {
    render(<ReviewQueue {...base} items={[item({ trust_state: "insufficient_evidence" })]} />);
    expect(within(screen.getByTestId("row-t-0001")).getByText("Insufficient Evidence")).toBeInTheDocument();
  });

  it("colours an overdue SLA differently and counts time since the fetch", () => {
    const { rerender } = render(<ReviewQueue {...base} items={[item({ sla: { minutes: 120, due_at: "", seconds_left: 30, breached: false } })]} />);
    expect(screen.getByText("00:00:30").className).toMatch(/sla-urgent/);
    rerender(<ReviewQueue {...base} elapsedSeconds={90} items={[item({ sla: { minutes: 120, due_at: "", seconds_left: 30, breached: false } })]} />);
    const overdue = screen.getByText("-00:01:00");
    expect(overdue.className).toMatch(/sla-breached/);
  });

  it("selects a case on click and on Enter", () => {
    const onSelect = vi.fn();
    render(<ReviewQueue {...base} onSelect={onSelect} items={[item()]} />);
    fireEvent.click(screen.getByTestId("row-t-0001"));
    fireEvent.keyDown(screen.getByTestId("row-t-0001"), { key: "Enter" });
    expect(onSelect).toHaveBeenCalledTimes(2);
    expect(onSelect).toHaveBeenCalledWith("d-1");
  });

  it("reports filter changes upward without changing other filters", () => {
    const onFilters = vi.fn();
    render(<ReviewQueue {...base} onFilters={onFilters} filters={{ ...EMPTY_FILTERS, trust: "low" }} items={[item({ reason_codes: ["THIN_COHORT"] })]} />);
    fireEvent.change(screen.getByLabelText("Filter by risk band"), { target: { value: "high" } });
    expect(onFilters).toHaveBeenLastCalledWith({ risk: "high", trust: "low", reason: "", search: "" });
    fireEvent.change(screen.getByLabelText("Filter by reason code"), { target: { value: "THIN_COHORT" } });
    expect(onFilters).toHaveBeenLastCalledWith({ risk: "", trust: "low", reason: "THIN_COHORT", search: "" });
    fireEvent.change(screen.getByLabelText("Search transactions"), { target: { value: "t-00" } });
    expect(onFilters).toHaveBeenLastCalledWith({ risk: "", trust: "low", reason: "", search: "t-00" });
  });

  it("offers the reason codes present in the data", () => {
    render(<ReviewQueue {...base} items={[item({ reason_codes: ["THIN_COHORT"] }), item({ decision_id: "d-2", txn_id: "t-2", reason_codes: ["WIDE_INTERVAL", "THIN_COHORT"] })]} />);
    const options = within(screen.getByLabelText("Filter by reason code")).getAllByRole("option").map((o) => o.textContent);
    expect(options).toEqual(["All", "THIN_COHORT", "WIDE_INTERVAL"]);
  });

  it("explains an empty queue instead of showing a blank table", () => {
    render(<ReviewQueue {...base} items={[]} total={0} />);
    expect(screen.getByText(/No cases match/)).toBeInTheDocument();
  });

  it("does not claim an empty queue while still loading", () => {
    render(<ReviewQueue {...base} items={[]} total={0} loading />);
    expect(screen.queryByText(/No cases match/)).not.toBeInTheDocument();
  });
});
