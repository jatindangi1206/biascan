import type { AnalyzeResponse, BiasType } from "../types";
import { BIAS_COLORS, BIAS_LABELS } from "../types";

interface Props {
  result: AnalyzeResponse;
}

const BIAS_ORDER: BiasType[] = [
  "confirmation_bias",
  "certainty_inflation",
  "overgeneralisation",
  "framing_effect",
  "causal_inference_error",
];

export function ResultsPanel({ result }: Props) {
  const scoreValue = result.overall_bias_score * 10;
  const scoreLabel =
    scoreValue >= 7.5
      ? "Severe"
      : scoreValue >= 5.5
        ? "Concerning"
        : scoreValue >= 3.0
          ? "Moderate"
          : "Low";

  const counts = BIAS_ORDER.map((biasType) => ({
    biasType,
    count: result.annotations.filter((annotation) => annotation.bias_type === biasType)
      .length,
  })).filter((entry) => entry.count > 0);

  const severityCounts = result.annotations.reduce(
    (acc, a) => {
      acc[a.severity] = (acc[a.severity] ?? 0) + 1;
      return acc;
    },
    { high: 0, medium: 0, low: 0 } as Record<"high" | "medium" | "low", number>,
  );
  const compositionParts = (["high", "medium", "low"] as const)
    .filter((s) => severityCounts[s] > 0)
    .map((s) => `${severityCounts[s]} ${s}`);
  const uniqueTypes = counts.length;
  const composition = result.annotations.length
    ? `${compositionParts.join(", ")} · ${uniqueTypes} bias ${uniqueTypes === 1 ? "type" : "types"}`
    : null;

  // Score breakdown — must stay in sync with _overall_score in
  // backend/app/agents/orchestrator.py. Count drives the score; severity
  // and diversity add small bumps.
  const n = result.annotations.length;
  const flagCountPts = n === 0 ? 0 : 2.5 + 5.5 * (1 - 1 / (1 + n / 3));
  const severityPts = Math.min(
    1.5,
    0.4 * severityCounts.high + 0.15 * severityCounts.medium,
  );
  const diversityPts = 0.15 * Math.max(0, uniqueTypes - 1);

  return (
    <section className="summary-section">
      <div className="summary-metrics">
        <div className="metric-block">
          <p className="section-label">Bias score</p>
          <div className="score-line">
            <span className="score-value">{scoreValue.toFixed(1)}</span>
            <span className="score-context">/10 · {scoreLabel}</span>
          </div>
          {composition && <p className="score-composition">{composition}</p>}
        </div>

        <div className="metric-block metric-block-right">
          <p className="section-label">Flags</p>
          <div className="flag-value">{result.annotations.length}</div>
        </div>
      </div>

      {n > 0 && (
        <details className="score-breakdown">
          <summary>How is this calculated?</summary>
          <div className="breakdown-rows">
            <div className="breakdown-row">
              <span>
                Flag count ({n} flag{n === 1 ? "" : "s"})
              </span>
              <span className="breakdown-pts">+{flagCountPts.toFixed(2)}</span>
            </div>
            {severityPts > 0 && (
              <div className="breakdown-row">
                <span>
                  Severity ({severityCounts.high} high
                  {severityCounts.medium > 0 ? `, ${severityCounts.medium} medium` : ""})
                </span>
                <span className="breakdown-pts">+{severityPts.toFixed(2)}</span>
              </div>
            )}
            {diversityPts > 0 && (
              <div className="breakdown-row">
                <span>Bias-type spread ({uniqueTypes} types)</span>
                <span className="breakdown-pts">+{diversityPts.toFixed(2)}</span>
              </div>
            )}
            <div className="breakdown-row breakdown-total">
              <span>Total</span>
              <span className="breakdown-pts">{scoreValue.toFixed(1)} / 10</span>
            </div>
          </div>
          <p className="breakdown-note">
            Flag count is the main signal. Severity and bias-type spread add
            small bumps. Adding a flag never lowers the score.
          </p>
        </details>
      )}

      <div className="bias-list">
        {counts.map((entry) => (
          <div key={entry.biasType} className="bias-row">
            <span className="bias-name">
              <span
                className="bias-dot"
                style={{ background: BIAS_COLORS[entry.biasType].fg }}
              />
              {BIAS_LABELS[entry.biasType]}
            </span>
            <span>{entry.count}</span>
          </div>
        ))}
      </div>

      <details className="run-details">
        <summary>Run details</summary>
        <div className="detail-list">
          <p>
            Mode · <strong>{result.mode}</strong>
          </p>
          <p>
            Provider · <strong>{result.provider.provider}</strong>
          </p>
          <p>
            Model · <strong>{result.provider.model}</strong>
          </p>
          {result.agents.some((a) => a.error) &&
            result.agents
              .filter((a) => a.error)
              .map((agent) => (
                <p key={agent.agent} className="detail-error">
                  {agent.agent} · {agent.error}
                </p>
              ))}
        </div>
      </details>

      {result.annotations.length > 0 && (
        <details className="run-details">
          <summary>Flag details</summary>
          <div className="flag-detail-list">
            {result.annotations.map((annotation, index) => (
              <div key={`${annotation.agent_name}-${index}`} className="flag-card">
                <div className="flag-card-head">
                  <span className="flag-card-label">
                    {annotation.agent_name} · {BIAS_LABELS[annotation.bias_type]}
                  </span>
                  <span className="flag-card-meta">
                    {annotation.severity} ·{" "}
                    {(annotation.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="flag-card-quote">{annotation.flagged_text}</p>
                {annotation.clean_alternative && (
                  <p className="flag-card-copy">
                    Cleaner alternative · {annotation.clean_alternative}
                  </p>
                )}
                {annotation.false_positive_check && (
                  <p className="flag-card-copy">
                    False-positive check · {annotation.false_positive_check}
                  </p>
                )}
                <p className="flag-card-copy">
                  RAG check · {annotation.rag_check_needed ? "needed" : "not needed"}
                  {annotation.rag_query ? ` · ${annotation.rag_query}` : ""}
                </p>
                {Object.keys(annotation.extras).length > 0 && (
                  <details className="flag-card-extra">
                    <summary>Extra fields</summary>
                    <pre>{JSON.stringify(annotation.extras, null, 2)}</pre>
                  </details>
                )}
              </div>
            ))}
          </div>
        </details>
      )}
    </section>
  );
}
