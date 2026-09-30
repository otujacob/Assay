import type {
  ActionRequest, ActionResponse, BundleInfo, CaseView, Dashboard, DemoUser, QueueResponse, ValidationReport,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}

const USER_KEY = "assay.demoUser";
let currentUser = "analyst";
try {
  currentUser = localStorage.getItem(USER_KEY) ?? "analyst";
} catch {
  /* storage unavailable: keep the default */
}

export const getUser = (): string => currentUser;
export function setUser(key: string): void {
  currentUser = key;
  try {
    localStorage.setItem(USER_KEY, key);
  } catch {
    /* storage unavailable */
  }
}

async function call<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method,
    headers: { "Content-Type": "application/json", "X-Demo-User": currentUser },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    /* non-JSON error body */
  }
  if (!res.ok) {
    const detail = (data as { detail?: { code?: string; detail?: string } | string } | null)?.detail;
    const code = typeof detail === "object" && detail ? (detail.code ?? "error") : "error";
    const msg = typeof detail === "object" && detail ? (detail.detail ?? detail.code ?? res.statusText) : String(detail ?? res.statusText);
    throw new ApiError(res.status, code, msg);
  }
  return data as T;
}

export interface QueueFilters {
  risk_band?: string;
  trust_state?: string;
  reason_code?: string;
  search?: string;
  queue?: string;
  include_closed?: boolean;
}

const qs = (f: QueueFilters): string => {
  const p = new URLSearchParams();
  Object.entries(f).forEach(([k, v]) => {
    if (v !== undefined && v !== "" && v !== false) p.set(k, String(v));
  });
  const s = p.toString();
  return s ? `?${s}` : "";
};

export const api = {
  users: async (): Promise<DemoUser[]> => {
    const res = await fetch("/demo/users");
    if (!res.ok) throw new ApiError(res.status, "demo_users", "demo server not reachable");
    return res.json();
  },
  queue: (f: QueueFilters = {}) => call<QueueResponse>("GET", `/review/queue${qs(f)}`),
  getCase: (decisionId: string) => call<CaseView>("GET", `/review/cases/${decisionId}`),
  act: (decisionId: string, body: ActionRequest) =>
    call<ActionResponse>("POST", `/review/cases/${decisionId}/actions`, body),
  adjudicate: (decisionId: string, body: { final_decision: "approve" | "block"; rationale: string; notes?: string }) =>
    call<CaseView>("POST", `/review/cases/${decisionId}/adjudication`, body),
  dashboard: () => call<Dashboard>("GET", "/dashboard/summary"),
  reports: () => call<ValidationReport[]>("GET", "/validation/reports?limit=1"),
  bundles: () => call<BundleInfo[]>("GET", "/models/bundles"),
};
