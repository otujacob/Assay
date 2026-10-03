// Shapes returned by the Assay API (backend/src/assay/api/app.py). Kept by hand: the OpenAPI spec
// (PRD 26.1) is the contract, and these mirror it.

export type TrustState = "high" | "moderate" | "low" | "insufficient_evidence";
export type RiskBand = "low" | "medium" | "high";
export type ComponentStatus = "active" | "inactive" | "missing";
export type ComponentName = "conf" | "rel" | "exp" | "fam" | "drift" | "dq" | "hum";

export interface SlaInfo {
  minutes: number;
  due_at: string;
  seconds_left: number;
  breached: boolean;
}

export interface QueueItem {
  decision_id: string;
  txn_id: string;
  status: CaseStatus;
  queue: string;
  priority: "P1" | "P2" | "P3" | "P4";
  priority_score: number;
  blind: boolean;
  redacted: boolean;
  risk_band: RiskBand | null;
  trust_state: TrustState | null;
  reason_codes: string[];
  amount: number;
  currency: string;
  event_time: string;
  sla: SlaInfo;
}

export interface AuditRow {
  seq: number | null;
  time: string;
  actor: string;
  action: string;
  object: string;
  result: string;
  row_hash: string;
}

export interface AuditResponse {
  items: AuditRow[];
  matching: number;
  chain_ok: boolean;
}

export interface PolicyVersion {
  id: string;
  version: string;
  payload: { dq_gate_action: "request_human_review" | "hold"; automation_level: number };
  effective_from: string;
  proposed_by: string;
  status: "pending" | "approved";
  approved_by: string | null;
  approved_at: string | null;
}

export type CaseStatus = "open" | "escalated" | "decided" | "conflicted" | "adjudicated" | "superseded";

export interface QueueResponse {
  items: QueueItem[];
  matching: number;
  total_cases: number;
  sla_breached: number;
}

export interface TrustComponent {
  status: ComponentStatus;
  score: number | null;
  lo: number | null;
  hi: number | null;
  n: number;
}

export interface Trust {
  state: TrustState;
  mode: string;
  ti: number | null;
  ti_low: number | null;
  ti_high: number | null;
  reason_codes: string[];
  components: Partial<Record<ComponentName, TrustComponent>>;
  version_no: number;
}

export interface Decision {
  decision_id: string;
  txn_id: string;
  risk: number;
  risk_band: RiskBand;
  trust: Trust;
  recommendation: string;
  gate: string;
  queue: string | null;
  automation_level: number;
  bundle_id: string;
  policy_version: string;
  explanation_status: "computed" | "pending";
}

export interface Explanation {
  method: string;
  params: Record<string, unknown>;
  background_version: string;
  seed: number;
  attributions: Record<string, number>;
  stability: number | null;
  sensitivity: number | null;
  faithfulness: number | null;
  reproducible: boolean;
}

export interface ActionRecord {
  analyst: string;
  role: string;
  action: string;
  final_decision: "approve" | "block" | null;
  reason_code: string | null;
  at: string;
}

export interface CaseView {
  decision_id: string;
  txn_id: string;
  status: CaseStatus;
  final_decision: "approve" | "block" | null;
  queue: string;
  blind: boolean;
  redacted: boolean;
  priority: { score: number; label: string };
  sla: SlaInfo;
  transaction: {
    amount: number;
    currency: string;
    channel: string;
    event_time: string;
    country: string | null;
    merchant_id: string | null;
    customer: string | null;
    beneficiary: string | null;
  };
  outcome: { verified: "fraud" | "legitimate"; type: string } | null;
  decision: Decision | null;
  explanation: Explanation | null;
  actions: ActionRecord[];
  needs_second_review: boolean;
  queued: boolean;
}

export interface ActionRequest {
  action: "approve" | "block" | "escalate" | "request_review" | "override" | "unsure";
  reason_code?: string;
  override_to?: "approve" | "block";
  confidence?: number;
  checklist?: Record<string, boolean>;
  notes?: string;
  seconds_to_decision?: number;
}

export interface FeedbackScores {
  aas: number | null;
  fcs: number;
  lvs: number;
  crs: number | null;
  fqs: number;
  disposition: "accept" | "reject" | "defer";
  reason: string;
  formula_version: string;
  note: string;
}

export interface ActionResponse {
  action_id: string;
  status: CaseStatus;
  final_decision: "approve" | "block" | null;
  feedback: FeedbackScores;
  needs_second_review: boolean;
}

export interface RateEstimate {
  value: number | null;
  lo: number | null;
  hi: number | null;
  n: number;
  basis?: string;
  k?: number;
}

export interface Dashboard {
  as_of: string;
  pending_reviews: { count: number; total_cases: number; sla_breached: number; soonest_sla_seconds: number | null };
  high_trust_error_rate: RateEstimate & { basis: string };
  insufficient_evidence_rate: { value: number | null; n: number; previous: number | null; previous_n: number; window_days: number };
  feedback_acceptance: { accepted: number; rejected: number; deferred: number; acceptance_rate: number | null; note: string };
  trust_distribution: Record<string, number>;
  assessed_cases: number;
  model_drift: { status: "stable" | "alarm"; max_feature_drift: number };
  data_quality: { status: "healthy" | "degraded"; floor_share: number };
}

export interface Ci {
  value: number;
  lo: number | null;
  hi: number | null;
}

export interface ValidationReport {
  id: string;
  bundle_id: string;
  dataset_id: string;
  mode: string;
  n_cases: number;
  measures: {
    auroc: Ci;
    high_trust_error_rate: RateEstimate;
    low_trust_detection_rate: RateEstimate;
    false_confidence_rate: RateEstimate;
    scored_share: number;
    state_counts: Record<string, number>;
    coverage_curve: { coverage: number; error_rate: number; errors_kept: number }[];
  };
  baselines: Record<string, unknown> & {
    B1_distance_from_threshold: { auroc: Ci; trust_index_minus_baseline: { diff: number; lo: number; hi: number } };
    B2_ensemble_disagreement: { auroc: Ci; trust_index_minus_baseline: { diff: number; lo: number; hi: number } };
    B3_max_class_probability: { auroc: Ci; trust_index_minus_baseline: { diff: number; lo: number; hi: number } };
    B4_single_components: Record<string, { auroc: number }>;
  };
  ablations: Record<string, { auroc: number; delta_auroc: number; verdict: string }>;
  stress_results: Record<string, { test: string; passed: boolean | null; expected: string; status?: string }> | null;
  pass_criteria: {
    atce_supported_as_specified: boolean | null;
    trust_index_beats_B1_to_B3_with_ci_excluding_no_improvement: Record<string, boolean>;
    high_trust_error_rate_materially_below_overall: boolean | null;
  };
  limitations: string[];
  created_at?: string;
}

export interface BundleInfo {
  bundle_id: string;
  dataset_id: string;
  code_commit: string;
  status: string;
  artifact_sha256: string;
  champion: boolean;
  thresholds: { t_low: number; t_high: number } | null;
  metrics: { pr_auc_ensemble?: number; pr_auc_logistic_baseline?: number; test_rows?: number } | null;
  calibration: {
    ece_test?: number;
    reliability_test?: { n: number; mean_predicted: number; observed_rate: number }[];
  } | null;
  created_at: string | null;
  feature_set_version: string | null;
}

export interface DemoUser {
  key: string;
  label: string;
  roles: string[];
}
