import type { ComponentName, RiskBand, TrustState } from "../types";

export const pct = (x: number | null | undefined, digits = 1): string =>
  x === null || x === undefined ? "n/a" : `${(100 * x).toFixed(digits)}%`;

export const num = (x: number | null | undefined, digits = 2): string =>
  x === null || x === undefined ? "n/a" : x.toFixed(digits);

export const money = (amount: number, currency: string): string =>
  new Intl.NumberFormat("en-GB", { style: "currency", currency, maximumFractionDigits: 2 }).format(amount);

/** "01:18:32" for time left, "-02:10:05" for time overdue. */
export function formatSla(secondsLeft: number): string {
  const sign = secondsLeft < 0 ? "-" : "";
  const s = Math.abs(Math.round(secondsLeft));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const two = (n: number) => String(n).padStart(2, "0");
  return `${sign}${two(h)}:${two(m)}:${two(sec)}`;
}

export type SlaTone = "breached" | "urgent" | "soon" | "ok";
export function slaTone(secondsLeft: number, totalMinutes: number): SlaTone {
  if (secondsLeft < 0) return "breached";
  const frac = secondsLeft / (totalMinutes * 60);
  return frac < 0.25 ? "urgent" : frac < 0.5 ? "soon" : "ok";
}

export const formatTime = (iso: string): string => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : `${d.toISOString().slice(0, 10)} ${d.toISOString().slice(11, 16)}`;
};

export const TRUST_LABEL: Record<TrustState, string> = {
  high: "High",
  moderate: "Moderate",
  low: "Low",
  insufficient_evidence: "Insufficient Evidence",
};

export const RISK_LABEL: Record<RiskBand, string> = { low: "Low", medium: "Medium", high: "High" };

/** What a recommendation means to an analyst. Recommend-only: the system never acts (FR-23). */
export const RECOMMENDATION_LABEL: Record<string, string> = {
  approve: "Approve",
  approve_sampled_qa: "Approve, sampled for QA",
  block: "Block",
  escalate: "Escalate",
  request_human_review: "Request Human Review",
  request_human_review_priority: "Request Human Review (priority)",
  hold: "Hold",
};

export const COMPONENT_LABEL: Record<ComponentName, string> = {
  conf: "Model Confidence",
  rel: "Model Reliability",
  exp: "Explanation Reliability",
  fam: "Familiarity",
  drift: "Drift Stability",
  dq: "Data Quality",
  hum: "Human Evidence",
};

export const COMPONENT_ORDER: ComponentName[] = ["conf", "rel", "exp", "fam", "drift", "dq", "hum"];

export const GROUP_LABEL: Record<string, string> = {
  amount: "Amount vs baseline",
  velocity: "Transaction velocity",
  timing: "Time of day",
  channel: "Channel",
  counterparty: "New / shared beneficiary",
  device_location: "New device / location",
  completeness: "Missing fields",
};

export const REASON_LABEL: Record<string, string> = {
  MISSING_CRITICAL: "A critical component is unavailable",
  THIN_COHORT: "Too few matured outcomes for this cohort",
  DATA_QUALITY_FLOOR: "Input data quality is below the floor",
  UNFAMILIAR_PATTERN: "Unfamiliar pattern: unlike anything seen before",
  WIDE_INTERVAL: "Uncertainty interval is too wide to rely on",
  MODEL_TOO_NEW: "Model version has too few matured outcomes",
  STALE_REFERENCE: "Reference data is out of date",
  ATCE_UNAVAILABLE: "Trust engine unavailable",
};

/** Reason codes analysts choose from (PRD 10.5). The list is agreed with the fraud-operations
 *  adviser in Sprint 0; these are working defaults. */
export const ANALYST_REASON_CODES: { code: string; label: string }[] = [
  { code: "customer_confirmed_legitimate", label: "Customer confirmed the transaction" },
  { code: "customer_denied", label: "Customer denied the transaction" },
  { code: "pattern_matches_known_fraud", label: "Pattern matches known fraud" },
  { code: "device_or_location_anomaly", label: "Device or location anomaly" },
  { code: "beneficiary_suspicious", label: "Beneficiary looks suspicious" },
  { code: "amount_inconsistent_with_history", label: "Amount inconsistent with history" },
  { code: "model_unreliable_here", label: "Model seems unreliable on this case" },
  { code: "new_fraud_pattern", label: "Looks like a new fraud pattern" },
  { code: "insufficient_information", label: "Not enough information" },
];

export const CHECKLIST: { key: string; label: string }[] = [
  { key: "kyc_checked", label: "Customer profile reviewed" },
  { key: "customer_contacted", label: "Customer contacted" },
  { key: "device_reviewed", label: "Device and location reviewed" },
  { key: "beneficiary_reviewed", label: "Beneficiary reviewed" },
];

/** Same rule as the server (PRD 10.5, FR-26), mirrored to give early feedback. The server decides. */
export function needsReason(action: string, recommendation: string | null): boolean {
  if (action === "override") return true;
  const rec = recommendation === "approve" || recommendation === "approve_sampled_qa" ? "approve"
    : recommendation === "block" ? "block" : null;
  return (action === "approve" || action === "block") && rec !== null && action !== rec;
}
