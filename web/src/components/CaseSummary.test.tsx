import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import type { CaseSummaryView } from "../types";
import { CaseSummary } from "./CaseSummary";

const data: CaseSummaryView = {
  decision_id: "d-1", version: "summary-0", paragraph: "p", sources: ["prediction", "graph"],
  facts: [
    { text: "The model put the risk at 11.8%; its flagging threshold is 11.8%, so it flagged this case.", source: "prediction", ref: ["predictions.calibrated_risk"] },
    { text: "No other customer shares this case's device, IP address or beneficiary.", source: "graph", ref: ["graph.links"] },
  ],
  not_covered: ["what-ifs (on request)", "analyst notes"],
  note: "Written by fixed rules from the stored record: no language model was used and nothing left this system.",
};

beforeEach(() => vi.restoreAllMocks());

describe("CaseSummary", () => {
  it("is not worked out until asked, and says no AI model is involved", () => {
    const spy = vi.spyOn(api, "summary");
    render(<CaseSummary decisionId="d-1" />);
    expect(spy).not.toHaveBeenCalled();
    expect(screen.getByText(/no AI model is used and nothing leaves the system/)).toBeInTheDocument();
  });

  it("shows each fact with its source, what is not covered, and the server's note", async () => {
    vi.spyOn(api, "summary").mockResolvedValue(data);
    render(<CaseSummary decisionId="d-1" />);
    fireEvent.click(screen.getByRole("button", { name: "Summarise this case" }));
    const list = await screen.findByTestId("summary-facts");
    expect(list.querySelectorAll("li")).toHaveLength(2);
    expect(list).toHaveTextContent("flagged this case.");
    expect(list).toHaveTextContent("model score");
    expect(list).toHaveTextContent("linked entities");
    expect(screen.getByTestId("summary-not-covered")).toHaveTextContent("what-ifs (on request); analyst notes");
    expect(screen.getByTestId("summary-note")).toHaveTextContent("no language model was used");
  });

  it("shows the server's refusal, for example on a blind case", async () => {
    vi.spyOn(api, "summary").mockRejectedValue(new Error("this case is under blind review until you record a decision"));
    render(<CaseSummary decisionId="d-1" />);
    fireEvent.click(screen.getByRole("button", { name: "Summarise this case" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("under blind review");
  });
});
