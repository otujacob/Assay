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

const bandColour = (low: number | null): string =>
  low === null ? "var(--grey)" : low >= 70 ? "var(--green)" : low >= 40 ? "var(--amber)" : "var(--red)";

/**
 * AI Trust Index gauge. Insufficient evidence is shown as such, with its reasons, and NEVER as a
 * number or a zero (PRD 4.6, FR-25): the system does not manufacture a score to fill a gap.
 */
export function TrustGauge({ trust }: { trust: Trust }) {
  const insufficient = trust.state === "insufficient_evidence" || trust.ti === null;
  const low = trust.ti_low;
  return (
    <div className="gauge" data-testid="trust-gauge">
      <div className="gauge-head">
        <span className="card-title">AI Trust Index</span>
        <span className="mode-tag">{trust.mode === "provisional" ? "Provisional Mode" : "Calibrated Mode"}</span>
      </div>
      <div className="gauge-body">
        <svg viewBox="0 0 160 100" className="gauge-svg" role="img"
             aria-label={insufficient ? "Trust Index: insufficient evidence" : `Trust Index ${Math.round(trust.ti ?? 0)} out of 100`}>
          <path d={arcPath(0, 100)} className="gauge-track" />
          {/* band boundaries from the defaults in PRD 5.5: placeholders, not validated */}
          {[40, 70].map((v) => {
            const [x0, y0] = at(v, R - 8);
            const [x1, y1] = at(v, R + 8);
            return <line key={v} x1={x0} y1={y0} x2={x1} y2={y1} className="gauge-tick" />;
          })}
          {insufficient ? (
            <path d={arcPath(0, 100)} className="gauge-none" />
          ) : (
            <>
              <path d={arcPath(trust.ti_low ?? 0, trust.ti_high ?? 0)} className="gauge-interval" style={{ stroke: bandColour(low) }} />
              <path d={arcPath(0, trust.ti ?? 0)} className="gauge-value" style={{ stroke: bandColour(low) }} />
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
                {Math.round(trust.ti ?? 0)}
                <span className="gauge-of"> / 100</span>
              </div>
              <div className="muted small" title="Uncertainty interval. Bands use the lower bound (PRD 5.5).">
                Interval {Math.round(trust.ti_low ?? 0)} – {Math.round(trust.ti_high ?? 0)}
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
