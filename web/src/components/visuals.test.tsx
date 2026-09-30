import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { insufficientTrust, trust } from "../test/fixtures";
import { ComponentBars } from "./ComponentBars";
import { ShapDrivers, stabilityLabel } from "./ShapDrivers";
import { TrustGauge, arcPath } from "./TrustGauge";

describe("TrustGauge", () => {
  it("shows the score, the interval and the mode for a scored case", () => {
    render(<TrustGauge trust={trust()} />);
    const g = screen.getByTestId("trust-gauge");
    expect(within(g).getByText("76")).toBeInTheDocument();
    expect(within(g).getByText(/Interval 62 – 79/)).toBeInTheDocument();
    expect(within(g).getByText("Provisional Mode")).toBeInTheDocument();
    expect(within(g).getByText("Moderate")).toBeInTheDocument();
  });

  it("shows Insufficient evidence as such, with reasons, and never as a number or zero (FR-25)", () => {
    render(<TrustGauge trust={insufficientTrust()} />);
    const g = screen.getByTestId("trust-gauge");
    expect(within(g).getByText("No score")).toBeInTheDocument();
    expect(within(g).getAllByText(/Insufficient evidence/i).length).toBeGreaterThan(0);
    expect(g.textContent).not.toMatch(/\b0\s*\/\s*100/);
    expect(g.textContent).not.toMatch(/Interval/);
    expect(within(g).getByText("THIN_COHORT")).toBeInTheDocument();
    expect(within(g).getByText("UNFAMILIAR_PATTERN")).toBeInTheDocument();
    expect(within(g).getByText(/Too few matured outcomes/)).toBeInTheDocument();
    expect(within(g).getByRole("img")).toHaveAttribute("aria-label", "Trust Index: insufficient evidence");
  });

  it("notes when the assessment was updated after explanation testing", () => {
    render(<TrustGauge trust={trust({ version_no: 2 })} />);
    expect(screen.getByText(/Assessment version 2/)).toBeInTheDocument();
  });

  it("draws arcs from the left (0) over the top to the right (100)", () => {
    expect(arcPath(0, 100)).toMatch(/^M 20\.00 80\.00 A 60 60 0 0 1 140\.00 80\.00$/);
    expect(arcPath(50, 50)).toMatch(/^M 80\.00 20\.00 A 60 60 0 0 1 80\.00 20\.00$/);
  });
});

describe("ComponentBars", () => {
  it("labels an inactive component Inactive and a missing one Missing, and shows Wilson evidence", () => {
    const t = trust();
    t.components.exp = { status: "missing", score: null, lo: null, hi: null, n: 0 };
    render(<ComponentBars trust={t} />);
    expect(within(screen.getByTestId("comp-hum")).getByText("Inactive")).toBeInTheDocument();
    expect(within(screen.getByTestId("comp-exp")).getByText("Missing")).toBeInTheDocument();
    expect(within(screen.getByTestId("comp-rel")).getByText(/Wilson, n=120/)).toBeInTheDocument();
    expect(within(screen.getByTestId("comp-conf")).getByText("0.80")).toBeInTheDocument();
  });

  it("lists all seven components in the PRD order", () => {
    render(<ComponentBars trust={trust()} />);
    const labels = screen.getAllByTestId(/^comp-/).map((e) => e.textContent ?? "");
    expect(labels).toHaveLength(7);
    expect(labels[0]).toContain("Model Confidence");
    expect(labels[6]).toContain("Human Evidence");
  });
});

describe("ShapDrivers", () => {
  const expl = {
    method: "treeshap-groups", params: {}, background_version: "b", seed: 0,
    attributions: { velocity: 0.1, amount: 2.4, device_location: -0.9 },
    stability: 0.6, sensitivity: 0.8, faithfulness: 0.7, reproducible: true,
  };

  it("orders drivers by absolute size and signs them", () => {
    render(<ShapDrivers explanation={expl} pending={false} />);
    const rows = screen.getAllByTestId(/^shap-(?!drivers|pending)/);
    expect(rows.map((r) => r.getAttribute("data-testid"))).toEqual(["shap-amount", "shap-device_location", "shap-velocity"]);
    expect(within(rows[0]).getByText("+2.40")).toBeInTheDocument();
    expect(within(rows[1]).getByText("-0.90")).toBeInTheDocument();
    expect(screen.getByText(/Moderate stability/)).toBeInTheDocument();
    expect(screen.getByText(/A stable, faithful explanation can still explain a wrong prediction/)).toBeInTheDocument();
  });

  it("explains that a pending explanation caps the trust index", () => {
    render(<ShapDrivers explanation={null} pending />);
    expect(screen.getByTestId("shap-pending")).toHaveTextContent(/capped below High/);
  });

  it("flags a non-reproducible explanation", () => {
    render(<ShapDrivers explanation={{ ...expl, reproducible: false }} pending={false} />);
    expect(screen.getByText(/NOT reproducible/)).toBeInTheDocument();
  });

  it("labels stability bands", () => {
    expect(stabilityLabel(0.9).text).toBe("High stability");
    expect(stabilityLabel(0.7).text).toBe("Moderate stability");
    expect(stabilityLabel(0.3).text).toBe("Low stability");
    expect(stabilityLabel(null).text).toBe("Unavailable");
  });
});
