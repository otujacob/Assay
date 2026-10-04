import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { insufficientTrust, trust } from "../test/fixtures";
import { TrustGauge } from "./TrustGauge";

const calibrated = (over = {}) => trust({ mode: "calibrated", state: "high", ti: 99.4, ti_low: 99.1, ti_high: 99.7, ...over });
const strokes = () => [...document.querySelectorAll<SVGPathElement>(".gauge-value, .gauge-interval")].map((p) => p.style.stroke);

describe("TrustGauge in calibrated mode", () => {
  it("shows a percentage to one decimal and says what the number means", () => {
    render(<TrustGauge trust={calibrated({ ti: 98.74, ti_low: 97.86, ti_high: 99.42, state: "moderate" })} />);
    const g = screen.getByTestId("trust-gauge");
    expect(g).toHaveTextContent("98.7%");
    expect(g).toHaveTextContent("Estimated chance this recommendation is correct");
    expect(g).toHaveTextContent("Interval 97.9 – 99.4%");
    expect(g).toHaveTextContent("Calibrated Mode");
    expect(g).not.toHaveTextContent("/ 100");
  });

  it("does not collapse close scores or their interval into the same whole number", () => {
    const { rerender } = render(<TrustGauge trust={calibrated({ ti: 98.6, ti_low: 98.0, ti_high: 99.1 })} />);
    const a = screen.getByTestId("trust-gauge").textContent;
    rerender(<TrustGauge trust={calibrated({ ti: 99.3, ti_low: 98.8, ti_high: 99.7 })} />);
    expect(screen.getByTestId("trust-gauge").textContent).not.toBe(a);
  });

  it("colours by the band the server decided, not by a cut-off on the number", () => {
    // 97 is far above the provisional 70, but in calibrated mode it is a Moderate case (about 3% expected error).
    const { unmount } = render(<TrustGauge trust={calibrated({ state: "moderate", ti: 97.8, ti_low: 97.2, ti_high: 98.3 })} />);
    expect(new Set(strokes())).toEqual(new Set(["var(--amber)"]));
    unmount();
    render(<TrustGauge trust={calibrated({ state: "low", ti: 93, ti_low: 91, ti_high: 95 })} />);
    expect(new Set(strokes())).toEqual(new Set(["var(--red)"]));
  });

  it("draws no band ticks, since the provisional 40 and 70 mean nothing for a probability", () => {
    render(<TrustGauge trust={calibrated()} />);
    expect(document.querySelectorAll(".gauge-tick")).toHaveLength(0);
  });

  it("describes itself accurately to a screen reader", () => {
    render(<TrustGauge trust={calibrated({ ti: 98.74 })} />);
    expect(screen.getByRole("img")).toHaveAccessibleName("Estimated chance the recommendation is correct: 98.7 percent");
  });

  it("still shows Insufficient evidence as no score, never as a number", () => {
    render(<TrustGauge trust={{ ...insufficientTrust(), mode: "calibrated" }} />);
    const g = screen.getByTestId("trust-gauge");
    expect(g).toHaveTextContent("No score");
    expect(g).not.toHaveTextContent("%");
  });
});

describe("TrustGauge in provisional mode is unchanged", () => {
  it("keeps whole numbers out of 100, the band ticks, and says it is not a probability", () => {
    render(<TrustGauge trust={trust()} />);
    const g = screen.getByTestId("trust-gauge");
    expect(g).toHaveTextContent("76");
    expect(g).toHaveTextContent("/ 100");
    expect(g).toHaveTextContent("Interval 62 – 79");
    expect(document.querySelectorAll(".gauge-tick")).toHaveLength(2);
    expect(screen.getByText("Provisional Mode")).toHaveAttribute("title", expect.stringMatching(/not a probability/));
  });

  it("colours by band too, so a Moderate case is amber and a High one green", () => {
    const { unmount } = render(<TrustGauge trust={trust({ state: "moderate" })} />);
    expect(new Set(strokes())).toEqual(new Set(["var(--amber)"]));
    unmount();
    render(<TrustGauge trust={trust({ state: "high", ti: 88, ti_low: 80, ti_high: 90 })} />);
    expect(new Set(strokes())).toEqual(new Set(["var(--green)"]));
  });
});
