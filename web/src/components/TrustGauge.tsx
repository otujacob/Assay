import type { Trust } from "../types";
import { REASON_LABEL, TRUST_LABEL } from "../lib/format";
import { TrustPill } from "./Pill";

const CX = 80;
const CY = 80;
const R = 60;

/** Point on the semicircle for a value in [0, 100]: 0 at the left, 100 at the right, over the top. */
const at = (v: number, r = R): [number, number] => {
  const t = Math.PI * (1 - Math.min(100, Math.max(0, v)) / 100);
  return [CX + r * Math.cos(t), CY - r * Math.sin(t)];
};

export function arcPath(v0: number, v1: number, r = R): string {
  const [x0, y0] = at(v0, r);
  const [x1, y1] = at(v1, r);
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 0 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

/** Colour follows the case's BAND, which the server decided, not a cut-off applied to the number here: in
 *  calibrated mode the number is a probability and a score of 97 is Moderate, not "above 70". */
const bandColour = (state: Trust["state"]): string =>
  state === "high" ? "var(--green)" : state === "moderate" ? "var(--amber)" : state === "low" ? "var(--red)" : "var(--grey)";

const MODE_HELP = {
  provisional: "A ranking of how much evidence supports the recommendation. It is not a probability.",
  calibrated: "Among similar past cases, the recommendation was correct about this often.",
} as const;

/**
 * AI Trust Index gauge. Insufficient evidence is shown as such, with its reasons, and NEVER as a
 * number or a zero (PRD 4.6, FR-25): the system does not manufacture a score to fill a gap.
 */
export function TrustGauge({ trust }: { trust: Trust }) {
  const insufficient = trust.state === "insufficient_evidence" || trust.ti === null;
  const calibrated = trust.mode === "calibrated";
  const colour = bandColour(trust.state);
  // A calibrated score is a probability near 100, so whole numbers would hide the differences that matter.
  const fmt = (v: number | null) => (calibrated ? (v ?? 0).toFixed(1) : String(Math.round(v ?? 0)));
  return (
    <div className="gauge" data-testid="trust-gauge">
      <div className="gauge-head">
        <span className="card-title">AI Trust Index</span>
        <span className="mode-tag" title={MODE_HELP[calibrated ? "calibrated" : "provisional"]}>
          {calibrated ? "Calibrated Mode" : "Provisional Mode"}
        </span>
      </div>
      <div className="gauge-body">
        <svg viewBox="0 0 160 100" className="gauge-svg" role="img"
             aria-label={insufficient ? "Trust Index: insufficient evidence"
                       : calibrated ? `Estimated chance the recommendation is correct: ${fmt(trust.ti)} percent`
                       : `Trust Index ${Math.round(trust.ti ?? 0)} out of 100`}>
          <path d={arcPath(0, 100)} className="gauge-track" />
          {/* provisional band boundaries (PRD 5.5 placeholders). Calibrated bands are set by tolerated error rates,
              which the gauge is not told, so it draws none rather than a wrong one. */}
          {(calibrated ? [] : [40, 70]).map((v) => {
            const [x0, y0] = at(v, R - 8);
            const [x1, y1] = at(v, R + 8);
            return <line key={v} x1={x0} y1={y0} x2={x1} y2={y1} className="gauge-tick" />;
          })}
          {insufficient ? (
            <path d={arcPath(0, 100)} className="gauge-none" />
          ) : (
            <>
              <path d={arcPath(trust.ti_low ?? 0, trust.ti_high ?? 0)} className="gauge-interval" style={{ stroke: colour }} />
              <path d={arcPath(0, trust.ti ?? 0)} className="gauge-value" style={{ stroke: colour }} />
            </>
          )}
        </svg>
        <div className="gauge-read">
          {insufficient ? (
            <>
              <div className="gauge-none-text">No score</div>
              <div className="muted small">Insufficient evidence</div>
            </>
          ) : (
            <>
              <div className="gauge-number">
                {fmt(trust.ti)}
                <span className="gauge-of">{calibrated ? "%" : " / 100"}</span>
              </div>
              {calibrated && <div className="small">Estimated chance this recommendation is correct</div>}
              <div className="muted small" title="Uncertainty interval. Bands use the lower bound (PRD 5.5).">
                Interval {fmt(trust.ti_low)} – {fmt(trust.ti_high)}{calibrated ? "%" : ""}
              </div>
            </>
          )}
        </div>
        <div className="gauge-band">
          <TrustPill state={trust.state} />
          <div className="muted tiny">{trust.state === "insufficient_evidence" ? "Human review required" : `Band: ${TRUST_LABEL[trust.state]}`}</div>
        </div>
      </div>
      {trust.reason_codes.length > 0 && (
        <div className="reasons">
          <div className="tiny muted">Reason codes</div>
          <div className="reason-list">
            {trust.reason_codes.map((c) => (
              <span key={c} className="reason-code" title={REASON_LABEL[c] ?? c}>{c}</span>
            ))}
          </div>
          {trust.reason_codes.map((c) => (
            <div key={c} className="tiny muted">{REASON_LABEL[c] ?? c}</div>
          ))}
        </div>
      )}
      {trust.version_no > 1 && <div className="tiny muted">Assessment version {trust.version_no} (updated when explanation testing completed)</div>}
    </div>
  );
}
