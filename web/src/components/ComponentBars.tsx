import type { ComponentName, Trust, TrustComponent } from "../types";
import { COMPONENT_LABEL, COMPONENT_ORDER, num } from "../lib/format";

const tone = (score: number): string => (score >= 0.75 ? "good" : score >= 0.5 ? "mid" : "bad");

function Row({ name, c }: { name: ComponentName; c: TrustComponent | undefined }) {
  const status = c?.status ?? "missing";
  return (
    <div className="comp-row" data-testid={`comp-${name}`}>
      <div className="comp-label">
        <span>{`${COMPONENT_ORDER.indexOf(name) + 1}. ${COMPONENT_LABEL[name]}`}</span>
      </div>
      <div className="comp-bar">
        {status === "active" && c?.score !== null && c?.score !== undefined ? (
          <>
            <div className={`comp-fill ${tone(c.score)}`} style={{ width: `${Math.round(c.score * 100)}%` }} />
            {c.lo !== null && c.hi !== null && c.hi > c.lo && (
              <div className="comp-interval" style={{ left: `${c.lo * 100}%`, width: `${(c.hi - c.lo) * 100}%` }}
                   title={`Interval ${num(c.lo)} to ${num(c.hi)}`} />
            )}
          </>
        ) : (
          <div className="comp-none" />
        )}
      </div>
      <div className="comp-value">
        {status === "active" && c?.score !== null && c?.score !== undefined ? (
          <>
            {num(c.score)}
            {name === "rel" && <span className="tiny muted"> (Wilson, n={c.n})</span>}
          </>
        ) : status === "inactive" ? (
          <span className="tag-inactive" title="Not enabled in the MVP: no verified analyst history yet. Weights renormalise (PRD 5.4).">Inactive</span>
        ) : (
          <span className="tag-missing" title="Expected but unavailable for this case. Penalised, not ignored (PRD 5.4).">Missing</span>
        )}
      </div>
    </div>
  );
}

export function ComponentBars({ trust }: { trust: Trust }) {
  return (
    <div className="card" data-testid="component-bars">
      <div className="card-title">Component Reliability</div>
      {COMPONENT_ORDER.map((n) => (
        <Row key={n} name={n} c={trust.components[n]} />
      ))}
      <div className="tiny muted pad-top">
        Bars show each component&rsquo;s score in [0, 1]. The thin overlay is its uncertainty interval. A missing component
        lowers the lower bound; an inactive one is removed and the weights renormalised.
      </div>
    </div>
  );
}
