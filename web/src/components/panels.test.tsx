import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { bundle, caseView, dashboard, insufficientTrust, report } from "../test/fixtures";
import { CaseDetails } from "./CaseDetails";
import { Governance } from "./Governance";
import { KpiCards } from "./KpiCards";

const noop = async () => {
  throw new Error("not used");
};

describe("CaseDetails", () => {
  const props = { canAct: true, canAdjudicate: false, onAct: noop as never, onAdjudicate: async () => {}, onBack: () => {} };

  it("shows score, trust, components, drivers and a recommend-only recommendation", () => {
    render(<CaseDetails view={caseView()} {...props} />);
    expect(screen.getByTestId("risk-card")).toHaveTextContent("0.82");
    expect(screen.getByTestId("trust-gauge")).toBeInTheDocument();
    expect(screen.getByTestId("component-bars")).toBeInTheDocument();
    expect(screen.getByTestId("shap-drivers")).toBeInTheDocument();
    expect(screen.getByTestId("recommendation")).toHaveTextContent(/Request Human Review/);
    expect(screen.getByTestId("recommendation")).toHaveTextContent(/recommend-only, nothing is done automatically/);
  });

  it("withholds score, trust and recommendation on a blind case but still shows the transaction", () => {
    render(<CaseDetails view={caseView({ blind: true, redacted: true, decision: null, explanation: null })} {...props} />);
    expect(screen.getByTestId("blind-notice")).toBeInTheDocument();
    expect(screen.queryByTestId("risk-card")).not.toBeInTheDocument();
    expect(screen.queryByTestId("trust-gauge")).not.toBeInTheDocument();
    expect(screen.queryByTestId("recommendation")).not.toBeInTheDocument();
    expect(screen.getByText(/1,250\.50/)).toBeInTheDocument();
    expect(screen.getByText("Blind review")).toBeInTheDocument();
  });

  it("shows an Insufficient-evidence case without inventing a score", () => {
    const v = caseView();
    v.decision!.trust = insufficientTrust();
    render(<CaseDetails view={v} {...props} />);
    expect(screen.getByText("No score")).toBeInTheDocument();
  });

  it("shows the adjudication form only to a senior analyst on a conflicted case", () => {
    const conflicted = caseView({ status: "conflicted" });
    const { rerender } = render(<CaseDetails view={conflicted} {...props} />);
    expect(screen.getByTestId("conflict-notice")).toHaveTextContent(/excluded from learning/);
    expect(screen.queryByLabelText("Adjudication rationale")).not.toBeInTheDocument();
    rerender(<CaseDetails view={conflicted} {...props} canAdjudicate />);
    expect(screen.getByLabelText("Adjudication rationale")).toBeInTheDocument();
  });

  it("requires a written rationale to adjudicate", async () => {
    const onAdjudicate = vi.fn().mockResolvedValue(undefined);
    render(<CaseDetails view={caseView({ status: "conflicted" })} {...props} canAdjudicate onAdjudicate={onAdjudicate} />);
    fireEvent.click(screen.getByRole("button", { name: /Adjudicate: block/ }));
    expect(screen.getByText(/needs a written rationale/)).toBeInTheDocument();
    expect(onAdjudicate).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Adjudication rationale"), { target: { value: "second evidence" } });
    fireEvent.click(screen.getByRole("button", { name: /Adjudicate: block/ }));
    await vi.waitFor(() => expect(onAdjudicate).toHaveBeenCalledWith("block", "second evidence"));
  });

  it("states a verified outcome separately from analyst decisions, and lists every decision", () => {
    const v = caseView({
      outcome: { verified: "fraud", type: "confirmed_fraud" },
      actions: [{ analyst: "u:a1", role: "analyst", action: "approve", final_decision: "approve", reason_code: "x", at: "2026-09-30T12:00:00Z" },
                { analyst: "u:a2", role: "analyst", action: "block", final_decision: "block", reason_code: "y", at: "2026-09-30T12:05:00Z" }],
      needs_second_review: true,
    });
    render(<CaseDetails view={v} {...props} />);
    expect(screen.getByText(/Verified outcome/)).toHaveTextContent(/fraud/);
    expect(within(screen.getByTestId("action-history")).getAllByText(/u:a/)).toHaveLength(2);
    expect(screen.getByText(/Every decision is kept/)).toBeInTheDocument();
    expect(screen.getByText(/goes to second review/)).toBeInTheDocument();
  });

  it("calls onBack", () => {
    const onBack = vi.fn();
    render(<CaseDetails view={caseView()} {...props} onBack={onBack} />);
    fireEvent.click(screen.getByRole("button", { name: /Back to Queue/ }));
    expect(onBack).toHaveBeenCalled();
  });
});

describe("KpiCards", () => {
  it("does not invent a high-trust error rate when there are no matured outcomes", () => {
    render(<KpiCards dash={dashboard()} elapsedSeconds={0} />);
    expect(screen.getByText("Not yet available")).toBeInTheDocument();
    expect(screen.getByText(/no matured verified outcomes yet/)).toBeInTheDocument();
    expect(screen.queryByText("Matured Outcomes")).not.toBeInTheDocument();
  });

  it("shows the rate with its interval and evidence count once outcomes exist", () => {
    render(<KpiCards elapsedSeconds={0} dash={dashboard({ high_trust_error_rate: { value: 0.0038, lo: 0.0019, hi: 0.0074, n: 2124, basis: "matured verified outcomes" } })} />);
    expect(screen.getByText(/0\.38%/)).toBeInTheDocument();
    expect(screen.getByText(/95% CI 0\.19% to 0\.74%/)).toBeInTheDocument();
    expect(screen.getByText(/n=2124/)).toBeInTheDocument();
  });

  it("shows pending reviews, the SLA breach count and a comparison only when one exists", () => {
    render(<KpiCards elapsedSeconds={0} dash={dashboard({ insufficient_evidence_rate: { value: 0.1, n: 50, previous: 0.06, previous_n: 40, window_days: 30 } })} />);
    expect(screen.getByText("31")).toBeInTheDocument();
    expect(screen.getByText("4 past SLA")).toBeInTheDocument();
    expect(screen.getByText(/▲ 4\.0% vs previous 30 days/)).toBeInTheDocument();
  });

  it("labels the most overdue case instead of a negative 'next SLA'", () => {
    render(<KpiCards elapsedSeconds={0} dash={dashboard({ pending_reviews: { count: 2, total_cases: 2, sla_breached: 2, soonest_sla_seconds: -3600 } })} />);
    expect(screen.getByText(/most overdue 01:00:00/)).toBeInTheDocument();
  });

  it("says why the figures are missing for a role that cannot see them", () => {
    render(<KpiCards dash={null} elapsedSeconds={0} />);
    expect(screen.getByText(/available to the manager role/)).toBeInTheDocument();
  });
});

describe("Governance", () => {
  it("labels everything as synthetic and shows the verdict as NOT supported when the criteria fail", () => {
    render(<Governance report={report()} bundle={bundle()} />);
    expect(screen.getByText("Synthetic data")).toBeInTheDocument();
    expect(screen.getByText("Not supported as specified")).toBeInTheDocument();
    expect(screen.getByText(/✗ beats B3/)).toBeInTheDocument();
    expect(screen.getByText(/✓ beats B1/)).toBeInTheDocument();
    expect(screen.getByText(/criteria are not relaxed when they fail/)).toBeInTheDocument();
  });

  it("shows the baseline comparison, the ablation with its verdicts, and the calibration figure", () => {
    render(<Governance report={report()} bundle={bundle()} />);
    expect(within(screen.getByTestId("baseline-bars")).getByText("0.808")).toBeInTheDocument();
    const ab = screen.getByTestId("ablation-table");
    expect(within(ab).getByText(/IMPROVES discrimination/)).toBeInTheDocument();
    expect(within(ab).getByText("+0.033")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /calibration reliability curve/i })).toBeInTheDocument();
    expect(screen.getByText(/ECE 0\.0050/)).toBeInTheDocument();
  });

  it("switches tabs to performance and bundle lineage", () => {
    render(<Governance report={report()} bundle={bundle()} />);
    fireEvent.click(screen.getByRole("tab", { name: "Performance & Calibration" }));
    expect(screen.getByText("0.38%")).toBeInTheDocument();
    expect(screen.getByText("not testable")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Model Bundle & Lineage" }));
    expect(screen.getByTestId("bundle-card")).toHaveTextContent("b-1");
    expect(screen.getByTestId("bundle-card")).toHaveTextContent("Champion");
    expect(screen.getByText("0.0570")).toBeInTheDocument();
  });

  it("shows one clear message, not empty cards, when a role has no governance access", () => {
    render(<Governance report={null} bundle={null} />);
    expect(screen.getByText(/available to the auditor, model-risk approver and manager roles/)).toBeInTheDocument();
    expect(screen.queryByText(/No validation report stored/)).not.toBeInTheDocument();
  });
});
