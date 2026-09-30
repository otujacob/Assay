import type { ReactNode } from "react";
import type { RiskBand, TrustState } from "../types";
import { RISK_LABEL, TRUST_LABEL } from "../lib/format";

export type Tone = "red" | "amber" | "green" | "blue" | "grey" | "purple";

export function Pill({ tone, children, title }: { tone: Tone; children: ReactNode; title?: string }) {
  return (
    <span className={`pill pill-${tone}`} title={title}>
      {children}
    </span>
  );
}

const TRUST_TONE: Record<TrustState, Tone> = { high: "green", moderate: "amber", low: "red", insufficient_evidence: "grey" };
const RISK_TONE: Record<RiskBand, Tone> = { low: "amber", medium: "amber", high: "red" };

export const TrustPill = ({ state }: { state: TrustState }) => (
  <Pill tone={TRUST_TONE[state]}>{TRUST_LABEL[state]}</Pill>
);

export const RiskPill = ({ band }: { band: RiskBand }) => <Pill tone={RISK_TONE[band]}>{RISK_LABEL[band]}</Pill>;

export const PriorityPill = ({ label }: { label: string }) => (
  <span className={`prio prio-${label.toLowerCase()}`}>{label}</span>
);
