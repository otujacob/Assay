import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import type { GraphLink, GraphView } from "../types";
import { LinkedEntities } from "./LinkedEntities";

const link = (over: Partial<GraphLink> = {}): GraphLink => ({
  other_customer: "c-r1", relationship: "SHARES_BENEFICIARY", via_kind: "beneficiary", via: "b-ring", confidence: 0.37,
  observations: 4, first_seen: "2025-01-02T00:00:00+00:00", last_seen: "2025-01-05T00:00:00+00:00", fraud_known: false,
  flagged_wrong: false, ...over,
});
const view = (over: Partial<GraphView> = {}): GraphView => ({
  decision_id: "d-1", customer: "c-new", as_of: "2025-01-11T00:00:02+00:00", graph_version: "graph-0",
  entities: [{ relationship: "PAID", kind: "beneficiary", node: "b-ring", other_customers: 3, degree: 3, hub: false }],
  links: [link(), link({ other_customer: "c-r2", fraud_known: true, confidence: 0.4 })], links_total: 2,
  group: { size: 3, links: 2, fraud_linked_members: 1 }, features: {}, reliability: "ok", reliability_note: null,
  note: "Shared infrastructure is evidence of a connection, not of wrongdoing. This view must not be shared with a customer.",
  ...over,
});

beforeEach(() => vi.restoreAllMocks());

describe("LinkedEntities", () => {
  it("does not load the graph until asked", () => {
    const spy = vi.spyOn(api, "graph");
    render(<LinkedEntities decisionId="d-1" canFlag />);
    expect(spy).not.toHaveBeenCalled();
    expect(screen.getByText(/connections, not guilt/)).toBeInTheDocument();
  });

  it("shows the entities, the group, each link with its confidence, and flags a confirmed fraud", async () => {
    vi.spyOn(api, "graph").mockResolvedValue(view());
    render(<LinkedEntities decisionId="d-1" canFlag />);
    fireEvent.click(screen.getByRole("button", { name: "Show linked entities" }));
    expect(await screen.findByTestId("graph-entities")).toHaveTextContent("used by 3 other customers");
    expect(screen.getByTestId("graph-group")).toHaveTextContent("Connected group of 3 customers (2 links; 1 with a confirmed fraud known at the time)");
    const rows = screen.getAllByTestId("graph-link");
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent("c-r1");
    expect(rows[0]).toHaveTextContent("37%");
    expect(within(rows[1]).getByText("confirmed fraud known")).toBeInTheDocument();
    expect(within(rows[0]).queryByText("confirmed fraud known")).not.toBeInTheDocument();
  });

  it("always shows the server's caveat and the as-of time, and never calls confidence a fraud probability", async () => {
    vi.spyOn(api, "graph").mockResolvedValue(view());
    render(<LinkedEntities decisionId="d-1" canFlag />);
    fireEvent.click(screen.getByRole("button", { name: "Show linked entities" }));
    expect(await screen.findByTestId("graph-note")).toHaveTextContent("must not be shared with a customer");
    expect(screen.getByText(/As the graph stood at/)).toBeInTheDocument();
    expect(screen.queryByText(/probability of fraud/i)).not.toBeInTheDocument();
  });

  it("says plainly when nothing is linked", async () => {
    vi.spyOn(api, "graph").mockResolvedValue(view({ links: [], links_total: 0, group: { size: 1, links: 0, fraud_linked_members: 0 },
      entities: [{ relationship: "USED_DEVICE", kind: "device", node: "d-solo", other_customers: 0, degree: 0, hub: false }] }));
    render(<LinkedEntities decisionId="d-1" canFlag />);
    fireEvent.click(screen.getByRole("button", { name: "Show linked entities" }));
    expect(await screen.findByTestId("graph-none")).toBeInTheDocument();
    expect(screen.getByTestId("graph-entities")).toHaveTextContent("not used by anyone else");
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("marks a widely shared node as a weak link", async () => {
    vi.spyOn(api, "graph").mockResolvedValue(view({ links: [], links_total: 0,
      entities: [{ relationship: "ACCESSED_FROM", kind: "IP address", node: "ip-cafe", other_customers: 70, degree: 70, hub: true }] }));
    render(<LinkedEntities decisionId="d-1" canFlag />);
    fireEvent.click(screen.getByRole("button", { name: "Show linked entities" }));
    expect(await screen.findByText("widely shared")).toBeInTheDocument();
  });

  it("warns when the view leans on weak links, and says when only the strongest are shown", async () => {
    vi.spyOn(api, "graph").mockResolvedValue(view({ reliability: "weak", reliability_note: "This view leans on weak links (confidence below 0.2).", links_total: 45 }));
    render(<LinkedEntities decisionId="d-1" canFlag />);
    fireEvent.click(screen.getByRole("button", { name: "Show linked entities" }));
    expect(await screen.findByTestId("graph-reliability")).toHaveTextContent("weak links");
    expect(screen.getByText(/Showing the 2 strongest of 45 links/)).toBeInTheDocument();
  });

  it("an analyst can flag a link wrong, with a reason, and it is sent against the other customer's own relationship", async () => {
    vi.spyOn(api, "graph").mockResolvedValue(view());
    const flag = vi.spyOn(api, "flagEdge").mockResolvedValue({ flagged: true });
    render(<LinkedEntities decisionId="d-1" canFlag />);
    fireEvent.click(screen.getByRole("button", { name: "Show linked entities" }));
    await screen.findByTestId("graph-entities");
    fireEvent.click(screen.getAllByRole("button", { name: "Flag as wrong" })[0]);
    const send = screen.getByRole("button", { name: "Flag link" });
    expect(send).toBeDisabled(); // a reason is required
    fireEvent.change(screen.getByLabelText("Reason for flagging"), { target: { value: "a family account" } });
    fireEvent.click(send);
    await waitFor(() => expect(flag).toHaveBeenCalledWith({ relationship: "PAID", src: "c-r1", dst: "b-ring", reason: "a family account" }));
    expect(await screen.findByText("flagged wrong")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Flag as wrong" })).toHaveLength(1); // the other link can still be flagged
  });

  it("people who cannot act on a case cannot flag links, and a link already flagged cannot be flagged again", async () => {
    vi.spyOn(api, "graph").mockResolvedValue(view({ links: [link({ flagged_wrong: true }), link({ other_customer: "c-r2" })] }));
    render(<LinkedEntities decisionId="d-1" canFlag={false} />);
    fireEvent.click(screen.getByRole("button", { name: "Show linked entities" }));
    await screen.findByTestId("graph-entities");
    expect(screen.queryByRole("button", { name: "Flag as wrong" })).not.toBeInTheDocument();
    expect(screen.getByText("flagged wrong")).toBeInTheDocument();
  });

  it("shows the server's refusal when flagging fails, and the load error when the graph cannot be read", async () => {
    vi.spyOn(api, "graph").mockResolvedValueOnce(view()).mockRejectedValueOnce(new Error("forbidden"));
    vi.spyOn(api, "flagEdge").mockRejectedValue(new Error("no such relationship in this tenant's graph"));
    const { unmount } = render(<LinkedEntities decisionId="d-1" canFlag />);
    fireEvent.click(screen.getByRole("button", { name: "Show linked entities" }));
    await screen.findByTestId("graph-entities");
    fireEvent.click(screen.getAllByRole("button", { name: "Flag as wrong" })[0]);
    fireEvent.change(screen.getByLabelText("Reason for flagging"), { target: { value: "x" } });
    fireEvent.click(screen.getByRole("button", { name: "Flag link" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("no such relationship");
    unmount();
    render(<LinkedEntities decisionId="d-2" canFlag />);
    fireEvent.click(screen.getByRole("button", { name: "Show linked entities" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load linked entities: forbidden");
  });
});
