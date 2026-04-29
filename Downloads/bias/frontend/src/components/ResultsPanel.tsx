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
  const score = Math.round(result.overall_bias_score * 100);
  const scoreLabel =
    score >= 66 ? "High concern" : score >= 33 ? "Moderate concern" : "Low concern";

  const counts = BIAS_ORDER.map((biasType) => ({
    biasType,
    count: result.annotations.filter((annotation) => annotation.bias_type === biasType)
      .length,
  })).filter((entry) => entry.count > 0);

  return (
    <section className="summary-section">
      <div className="summary-metrics">
        <div className="metric-block">
          <p className="section-label">Bias score</p>
          <div className="score-line">
            <span className="score-value">{score}</span>
            <span className="score-context">/100 · {scoreLabel}</span>
          </div>
        </div>

        <div className="metric-block metric-block-right">
          <p className="section-label">Flags</p>
          <div className="flag-value">{result.annotations.length}</div>
        </div>
      </div>

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

      {result.warnings.length > 0 && (
        <div className="warning-stack">
          {result.warnings.map((warning, index) => (
            <p key={index} className="warning-copy">
              {warning}
            </p>
          ))}
        </div>
      )}

      <details className="run-details" open>
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
          {result.provider.base_url && (
            <p>
              Endpoint · <strong>{result.provider.base_url}</strong>
            </p>
          )}
          <p>
            Document · <strong>{result.document_id}</strong>
          </p>
          {result.agents.map((agent) => (
            <p key={agent.agent}>
              {agent.agent} · {BIAS_LABELS[agent.bias_type]} · {agent.raw_count} raw /{" "}
              {agent.kept_count} kept{agent.error ? ` · ${agent.error}` : ""}
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
