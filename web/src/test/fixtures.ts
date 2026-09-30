import type {
  CaseView, Dashboard, QueueItem, Trust, TrustComponent, ValidationReport, BundleInfo,
} from "../types";

const comp = (score: number, lo = score, hi = score, n = 1): TrustComponent => ({ status: "active", score, lo, hi, n });

export const trust = (over: Partial<Trust> = {}): Trust => ({
  state: "moderate", mode: "provisional", ti: 76, ti_low: 62, ti_high: 79, reason_codes: [], version_no: 1,
  components: {
    conf: comp(0.8, 0.7, 0.9), rel: comp(0.7, 0.6, 0.78, 120), exp: comp(0.6, 0.5, 0.7, 18),
    fam: comp(0.9), drift: comp(0.85), dq: comp(0.95), hum: { status: "inactive", score: null, lo: null, hi: null, n: 0 },
  },
  ...over,
});

export const insufficientTrust = (): Trust => trust({
  state: "insufficient_evidence", ti: null, ti_low: null, ti_high: null, reason_codes: ["THIN_COHORT", "UNFAMILIAR_PATTERN"],
  components: { conf: comp(0.4), rel: comp(0, 0, 0.3, 4), exp: { status: "missing", score: null, lo: null, hi: null, n: 0 } },
});

export const item = (over: Partial<QueueItem> = {}): QueueItem => ({
  decision_id: "d-1", txn_id: "t-0001", status: "open", queue: "standard", priority: "P1", priority_score: 0.9,
  blind: false, redacted: false, risk_band: "high", trust_state: "moderate", reason_codes: [], amount: 1250.5,
  currency: "GBP", event_time: "2026-09-30T12:17:00Z",
  sla: { minutes: 120, due_at: "2026-09-30T14:17:00Z", seconds_left: 4712, breached: false }, ...over,
});

export const caseView = (over: Partial<CaseView> = {}): CaseView => ({
  decision_id: "d-1", txn_id: "t-0001", status: "open", final_decision: null, queue: "standard", blind: false,
  redacted: false, priority: { score: 0.9, label: "P1" },
  sla: { minutes: 120, due_at: "2026-09-30T14:17:00Z", seconds_left: 4712, breached: false },
  transaction: { amount: 1250.5, currency: "GBP", channel: "card_present", event_time: "2026-09-30T12:17:00Z",
    country: "GB", merchant_id: "m-1", customer: "c-0…55", beneficiary: null },
  outcome: null,
  decision: {
    decision_id: "d-1", txn_id: "t-0001", risk: 0.82, risk_band: "high", trust: trust(),
    recommendation: "request_human_review", gate: "matrix", queue: null, automation_level: 0, bundle_id: "b-1",
    policy_version: "policy-0", explanation_status: "computed",
  },
  explanation: {
    method: "treeshap-groups", params: {}, background_version: "b-1", seed: 0,
    attributions: { amount: 2.4, counterparty: 1.1, device_location: -0.3, velocity: 0.1 },
    stability: 0.6, sensitivity: 0.8, faithfulness: 0.7, reproducible: true,
  },
  actions: [], needs_second_review: false, queued: true, ...over,
});

export const dashboard = (over: Partial<Dashboard> = {}): Dashboard => ({
  as_of: "2026-09-30T12:00:00Z",
  pending_reviews: { count: 31, total_cases: 99, sla_breached: 4, soonest_sla_seconds: 600 },
  high_trust_error_rate: { value: null, lo: null, hi: null, n: 0, basis: "no matured verified outcomes yet for High-trust cases" },
  insufficient_evidence_rate: { value: 0.086, n: 162, previous: null, previous_n: 0, window_days: 30 },
  feedback_acceptance: { accepted: 0, rejected: 3, deferred: 5, acceptance_rate: 0, note: "shadow" },
  trust_distribution: { high: 10 }, assessed_cases: 10,
  model_drift: { status: "stable", max_feature_drift: 0 }, data_quality: { status: "healthy", floor_share: 0 }, ...over,
});

const ci = (value: number) => ({ value, lo: value - 0.05, hi: value + 0.05 });
export const report = (): ValidationReport => ({
  id: "r-1", bundle_id: "b-1", dataset_id: "ds-1", mode: "provisional", n_cases: 3911,
  measures: {
    auroc: ci(0.808), high_trust_error_rate: { value: 0.0038, lo: 0.0019, hi: 0.0074, n: 2124 },
    low_trust_detection_rate: { value: 0.7, lo: 0.5, hi: 0.8, n: 47 }, false_confidence_rate: { value: 0.17, lo: 0.09, hi: 0.3, n: 47 },
    scored_share: 0.944, state_counts: {}, coverage_curve: [],
  },
  baselines: {
    B1_distance_from_threshold: { auroc: ci(0.671), trust_index_minus_baseline: { diff: 0.137, lo: 0.042, hi: 0.226 } },
    B2_ensemble_disagreement: { auroc: ci(0.733), trust_index_minus_baseline: { diff: 0.075, lo: 0.013, hi: 0.139 } },
    B3_max_class_probability: { auroc: ci(0.767), trust_index_minus_baseline: { diff: 0.042, lo: -0.025, hi: 0.099 } },
    B4_single_components: { conf: { auroc: 0.738 } },
  },
  ablations: { conf: { auroc: 0.759, delta_auroc: -0.049, verdict: "adds value" }, fam: { auroc: 0.841, delta_auroc: 0.033, verdict: "removing it IMPROVES discrimination" } },
  stress_results: { drift: { test: "synthetic feature drift", passed: true, expected: "" }, noisy: { test: "noisy labels", passed: null, expected: "", status: "not_testable" } },
  pass_criteria: {
    atce_supported_as_specified: false,
    trust_index_beats_B1_to_B3_with_ci_excluding_no_improvement: { B1_distance_from_threshold: true, B2_ensemble_disagreement: true, B3_max_class_probability: false },
    high_trust_error_rate_materially_below_overall: true,
  },
  limitations: ["Synthetic data."],
});

export const bundle = (): BundleInfo => ({
  bundle_id: "b-1", dataset_id: "0123456789abcdef", code_commit: "uncommitted", status: "candidate",
  artifact_sha256: "abcdef0123456789", champion: true, thresholds: { t_low: 0.019, t_high: 0.057 },
  metrics: { pr_auc_ensemble: 0.76, pr_auc_logistic_baseline: 0.69 },
  calibration: { ece_test: 0.005, reliability_test: [{ n: 10, mean_predicted: 0.01, observed_rate: 0.012 }, { n: 10, mean_predicted: 0.5, observed_rate: 0.48 }] },
  created_at: "2026-09-30T12:00:00Z", feature_set_version: "fs-2",
});
