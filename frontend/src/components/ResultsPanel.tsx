import type { AnalyzeResponse, BiasType } from "../types";
import { BIAS_COLORS, BIAS_LABELS } from "../types";
import { extractReasoning } from "./AnnotatedOutput";

interface Props {
  result: AnalyzeResponse;
  text: string;
}

const BIAS_ORDER: BiasType[] = [
  "confirmation_bias",
  "certainty_inflation",
  "overgeneralisation",
  "framing_effect",
  "causal_inference_error",
];

export function ResultsPanel({ result, text }: Props) {
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

  // Score breakdown — must stay in sync with Orchestrator._overall_score in
  // backend/app/agents/orchestrator.py.
  const n = result.annotations.length;
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const effectiveWords = Math.max(50, wordCount);
  const severityWeights = { high: 3, medium: 2, low: 1 } as const;
  const expectedImpact = result.annotations.reduce(
    (sum, annotation) => sum + annotation.confidence * severityWeights[annotation.severity],
    0,
  );
  const diversityBump = 0.15 * Math.max(0, uniqueTypes - 1);
  const totalImpact = expectedImpact + diversityBump;
  const density = n === 0 ? 0 : (totalImpact / effectiveWords) * 100;

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
                Expected impact ({n} flag{n === 1 ? "" : "s"})
              </span>
              <span className="breakdown-pts">{expectedImpact.toFixed(2)}</span>
            </div>
            {diversityBump > 0 && (
              <div className="breakdown-row">
                <span>Diversity bump ({uniqueTypes} types)</span>
                <span className="breakdown-pts">+{diversityBump.toFixed(2)}</span>
              </div>
            )}
            <div className="breakdown-row">
              <span>
                Density normalization ({effectiveWords} effective words
                {wordCount < 50 ? ", 50-word floor applied" : ""})
              </span>
              <span className="breakdown-pts">{density.toFixed(2)} / 100w</span>
            </div>
            <div className="breakdown-row">
              <span>Final curve (1 - e^(-0.15 x density))</span>
              <span className="breakdown-pts">{scoreValue.toFixed(1)} / 10</span>
            </div>
            <div className="breakdown-row">
              <span>Total weighted impact</span>
              <span className="breakdown-pts">{totalImpact.toFixed(2)}</span>
            </div>
            <div className="breakdown-row">
              <span>
                Severity weights ({severityCounts.high} high, {severityCounts.medium} medium,{" "}
                {severityCounts.low} low)
              </span>
              <span className="breakdown-pts">H=3 · M=2 · L=1</span>
            </div>
            <div className="breakdown-row breakdown-total">
              <span>Total</span>
              <span className="breakdown-pts">{scoreValue.toFixed(1)} / 10</span>
            </div>
          </div>
          <p className="breakdown-note">
            Expected impact sums `confidence × severity weight` for each flag
            using High=3, Medium=2, and Low=1. That impact is normalized per
            100 words with a 50-word minimum floor so longer documents are not
            unfairly penalized, then mapped onto an exponential curve for the
            final score.
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

      {result.agents.some((a) => a.reasoning || a.error) && (
        <details className="run-details" open={result.annotations.length === 0}>
          <summary>Agent reasoning</summary>
          <div className="agent-reasoning-list">
            {result.agents.map((agent) => (
              <div key={agent.agent} className="agent-reasoning-card">
                <div className="agent-reasoning-head">
                  <span className="agent-reasoning-name">
                    {agent.agent} · {BIAS_LABELS[agent.bias_type]}
                  </span>
                  <span className="agent-reasoning-status">
                    {agent.error
                      ? "error"
                      : agent.kept_count === 0
                        ? "no flags"
                        : `${agent.kept_count} flag${agent.kept_count === 1 ? "" : "s"}`}
                  </span>
                </div>
                {agent.error && (
                  <p className="agent-reasoning-error">{agent.error}</p>
                )}
                {!agent.error && agent.reasoning && (
                  <dl className="agent-reasoning-fields">
                    {Object.entries(agent.reasoning).map(([key, value]) => (
                      <div key={key} className="agent-reasoning-row">
                        <dt>{key.replace(/_/g, " ")}</dt>
                        <dd>{typeof value === "string" ? value : JSON.stringify(value)}</dd>
                      </div>
                    ))}
                  </dl>
                )}
                {!agent.error && !agent.reasoning && (
                  <p className="agent-reasoning-empty">
                    (no chain_of_thought returned by the model)
                  </p>
                )}
              </div>
            ))}
          </div>
        </details>
      )}

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
                {(() => {
                  const reason = extractReasoning(annotation.extras);
                  return reason ? (
                    <p className="flag-card-copy">Why · {reason}</p>
                  ) : null;
                })()}
                {annotation.clean_alternative && (
                  <p className="flag-card-copy">
                    Cleaner alternative · {annotation.clean_alternative}
                  </p>
                )}
                {annotation.rag_check_needed && (
                  <p className="flag-card-copy">
                    RAG check needed
                    {annotation.rag_query ? ` · ${annotation.rag_query}` : ""}
                  </p>
                )}
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
