import { describe, expect, it } from "vitest";
import { formatSla, formatTime, money, needsReason, num, pct, slaTone } from "./format";

describe("formatSla", () => {
  it("formats time left and time overdue", () => {
    expect(formatSla(4712)).toBe("01:18:32");
    expect(formatSla(0)).toBe("00:00:00");
    expect(formatSla(-7805)).toBe("-02:10:05");
    expect(formatSla(59.6)).toBe("00:01:00");
  });
});

describe("slaTone", () => {
  it("is breached once overdue, then urgent, soon, ok by the share of time left", () => {
    expect(slaTone(-1, 120)).toBe("breached");
    expect(slaTone(10 * 60, 120)).toBe("urgent"); // under 25% of 2h
    expect(slaTone(50 * 60, 120)).toBe("soon");
    expect(slaTone(100 * 60, 120)).toBe("ok");
  });
});

describe("needsReason (mirrors the server rule, PRD 10.5 / FR-26)", () => {
  it("always needs one for an override", () => {
    expect(needsReason("override", "approve")).toBe(true);
    expect(needsReason("override", null)).toBe(true);
  });
  it("needs one when approving or blocking against the recommendation", () => {
    expect(needsReason("block", "approve")).toBe(true);
    expect(needsReason("approve", "block")).toBe(true);
    expect(needsReason("block", "approve_sampled_qa")).toBe(true);
  });
  it("needs none when agreeing, or when the recommendation was review or escalate", () => {
    expect(needsReason("approve", "approve")).toBe(false);
    expect(needsReason("approve", "approve_sampled_qa")).toBe(false);
    expect(needsReason("block", "block")).toBe(false);
    expect(needsReason("approve", "request_human_review")).toBe(false);
    expect(needsReason("block", "escalate")).toBe(false);
    expect(needsReason("escalate", "approve")).toBe(false);
    expect(needsReason("unsure", "block")).toBe(false);
  });
  it("cannot judge disagreement on a hidden (blind) recommendation", () => {
    expect(needsReason("approve", null)).toBe(false);
  });
});

describe("number formatting", () => {
  it("shows n/a for missing values, never a fake zero", () => {
    expect(pct(null)).toBe("n/a");
    expect(pct(undefined)).toBe("n/a");
    expect(pct(0.0038, 2)).toBe("0.38%");
    expect(num(null)).toBe("n/a");
    expect(num(0.6)).toBe("0.60");
  });
  it("formats money and times", () => {
    expect(money(1250.5, "GBP")).toContain("1,250.50");
    expect(formatTime("2026-09-30T12:17:00Z")).toBe("2026-09-30 12:17");
    expect(formatTime("not a date")).toBe("not a date");
  });
});
