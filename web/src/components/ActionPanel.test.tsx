import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../api";
import type { ActionResponse } from "../types";
import { caseView } from "../test/fixtures";
import { ActionPanel } from "./ActionPanel";

const ok: ActionResponse = {
  action_id: "a-1", status: "decided", final_decision: "approve", needs_second_review: false,
  feedback: { aas: null, fcs: 0.6, lvs: 0.3, crs: null, fqs: 0.45, disposition: "defer", reason: "awaiting verified outcome or corroboration", formula_version: "fqs-0", note: "shadow" },
};

function view(recommendation: string, over = {}) {
  const v = caseView(over);
  if (v.decision) v.decision.recommendation = recommendation;
  return v;
}

describe("ActionPanel", () => {
  it("submits an agreeing decision without a reason code", async () => {
    const onSubmit = vi.fn().mockResolvedValue(ok);
    render(<ActionPanel view={view("approve")} canAct onSubmit={onSubmit} />);
    fireEvent.click(screen.getByRole("button", { name: /Approve/ }));
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toMatchObject({ action: "approve", confidence: 0.7 });
    expect(onSubmit.mock.calls[0][0]).not.toHaveProperty("reason_code");
    expect(await screen.findByTestId("action-result")).toHaveTextContent(/Case status: decided/);
    expect(screen.getByTestId("action-result")).toHaveTextContent(/shadow, not used for learning/);
  });

  it("refuses to send an override with no reason code, and says why (FR-26)", () => {
    const onSubmit = vi.fn();
    render(<ActionPanel view={view("block")} canAct onSubmit={onSubmit} />);
    fireEvent.click(screen.getByRole("button", { name: /Override AI/ }));
    fireEvent.click(screen.getByRole("button", { name: "Submit override" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/override needs a reason code/i);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("sends an override with its target, reason, checklist and notes", async () => {
    const onSubmit = vi.fn().mockResolvedValue(ok);
    render(<ActionPanel view={view("block")} canAct onSubmit={onSubmit} />);
    fireEvent.click(screen.getByRole("button", { name: /Override AI/ }));
    fireEvent.click(screen.getByLabelText("approve"));
    fireEvent.change(screen.getByLabelText("Reason code"), { target: { value: "customer_confirmed_legitimate" } });
    fireEvent.click(screen.getByLabelText("Customer contacted"));
    fireEvent.change(screen.getByLabelText("Notes"), { target: { value: "called the customer" } });
    fireEvent.click(screen.getByRole("button", { name: "Submit override" }));
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit.mock.calls[0][0]).toEqual({
      action: "override", override_to: "approve", reason_code: "customer_confirmed_legitimate",
      confidence: 0.7, checklist: { customer_contacted: true }, notes: "called the customer",
    });
  });

  it("needs a reason when disagreeing with the recommendation, but not when escalating", async () => {
    const onSubmit = vi.fn().mockResolvedValue(ok);
    render(<ActionPanel view={view("block")} canAct onSubmit={onSubmit} />);
    fireEvent.click(screen.getByRole("button", { name: /Approve/ }));
    expect(screen.getByRole("alert")).toHaveTextContent(/Disagreeing with the recommendation needs a reason code/);
    expect(onSubmit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Escalate/ }));
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
  });

  it("shows the server's error and keeps the form so nothing is lost", async () => {
    const onSubmit = vi.fn().mockRejectedValue(new ApiError(409, "already_decided", "this analyst has already decided this case"));
    render(<ActionPanel view={view("approve")} canAct onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("Notes"), { target: { value: "keep me" } });
    fireEvent.click(screen.getByRole("button", { name: /Approve/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("this analyst has already decided this case");
    expect(screen.getByLabelText("Notes")).toHaveValue("keep me");
  });

  it("explains blind review: the recommendation is withheld, so disagreement cannot be judged", async () => {
    const onSubmit = vi.fn().mockResolvedValue(ok);
    render(<ActionPanel view={caseView({ blind: true, redacted: true, decision: null, explanation: null })} canAct onSubmit={onSubmit} />);
    expect(screen.getByText(/you decide without the model.s score or recommendation/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Block/ }));
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
  });

  it("offers no decision controls to a role that cannot decide, or on a closed case", () => {
    const { rerender } = render(<ActionPanel view={view("approve")} canAct={false} onSubmit={vi.fn()} />);
    expect(screen.queryByRole("button", { name: /Approve/ })).not.toBeInTheDocument();
    expect(screen.getByText(/can view this case but not record decisions/)).toBeInTheDocument();
    rerender(<ActionPanel view={view("approve", { status: "adjudicated", final_decision: "block" })} canAct onSubmit={vi.fn()} />);
    expect(screen.queryByRole("button", { name: /Approve/ })).not.toBeInTheDocument();
    expect(screen.getByText(/adjudicated/)).toHaveTextContent(/block/);
  });

  it("warns when an override of a high-trust recommendation goes to second review", async () => {
    const onSubmit = vi.fn().mockResolvedValue({ ...ok, needs_second_review: true });
    render(<ActionPanel view={view("block")} canAct onSubmit={onSubmit} />);
    fireEvent.click(screen.getByRole("button", { name: /Override AI/ }));
    fireEvent.change(screen.getByLabelText("Reason code"), { target: { value: "customer_denied" } });
    fireEvent.click(screen.getByRole("button", { name: "Submit override" }));
    expect(await screen.findByTestId("action-result")).toHaveTextContent(/second review/);
  });
});
