import type { AnalyzeResponse, Annotation } from "../types";
import { AGENT_BY_BIAS, BIAS_COLORS, BIAS_LABELS } from "../types";

interface Props {
  result: AnalyzeResponse;
}

export function ResultsPanel({ result }: Props) {
  const score = result.overall_bias_score;
  const scoreLabel =
    score >= 0.66 ? "High concern" : score >= 0.33 ? "Moderate" : "Low";

  return (
    <div className="panel">
      <div className="panel-header">
        <p className="overline">03 — Summary</p>
        <h2 className="panel-title">Detection report</h2>
      </div>

      <div className="stats-row">
        <div className="stat">
          <span className="stat-num">{(score * 100).toFixed(0)}</span>
          <span className="stat-label">Bias score · {scoreLabel}</span>
        </div>
        <div className="stat">
          <span className="stat-num">{result.annotations.length}</span>
          <span className="stat-label">Annotations</span>
        </div>
        <div className="stat">
          <span className="stat-num">
            {result.annotations.filter((a) => a.conflict).length}
          </span>
          <span className="stat-label">Conflicts</span>
        </div>
        <div className="stat">
          <span className="stat-num">{result.mode}</span>
          <span className="stat-label">Mode</span>
        </div>
        <div className="stat">
          <span className="stat-num" style={{ fontSize: 12, fontFamily: "monospace" }}>
            {result.document_id}
          </span>
          <span className="stat-label">Document ID</span>
        </div>
      </div>

      {result.warnings.length > 0 && (
        <div className="warnings">
          {result.warnings.map((w, i) => (
            <div key={i} className="warning">
              <strong>Warning</strong> · {w}
            </div>
          ))}
        </div>
      )}

      <h3 className="subhead">Per-agent run</h3>
      <table className="tbl">
        <thead>
          <tr>
            <th>Agent</th>
            <th>Bias</th>
            <th>Prompt</th>
            <th>Raw → kept</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {result.agents.map((a) => (
            <tr key={a.agent}>
              <td><strong>{a.agent}</strong></td>
              <td>{BIAS_LABELS[a.bias_type]}</td>
              <td><code>{a.prompt_version}</code></td>
              <td>{a.raw_count} → {a.kept_count}</td>
              <td>{a.error ? <span className="err">{a.error}</span> : "ok"}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 className="subhead">Annotations</h3>
      {result.annotations.length === 0 ? (
        <p className="hint">No annotations above the confidence floor.</p>
      ) : (
        <ul className="ann-list">
          {result.annotations.map((a, i) => (
            <AnnotationCard key={i} a={a} />
          ))}
        </ul>
      )}
    </div>
  );
}

function AnnotationCard({ a }: { a: Annotation }) {
  const palette = BIAS_COLORS[a.bias_type];
  return (
    <li className="ann-card" style={{ borderLeftColor: palette.fg }}>
      <div className="ann-card-head">
        <span className="ann-tag" style={{ background: palette.bg, color: palette.fg }}>
          {AGENT_BY_BIAS[a.bias_type]} · {BIAS_LABELS[a.bias_type]}
        </span>
        <span className={`sev sev-${a.severity}`}>{a.severity}</span>
        <span className="ann-meta">{(a.confidence * 100).toFixed(0)}% confidence</span>
        {a.conflict && <span className="conflict-tag">conflict</span>}
      </div>
      <p className="ann-quote">"{a.flagged_text}"</p>
      {a.clean_alternative && (
        <p className="ann-clean">
          <strong>Cleaner:</strong> {a.clean_alternative}
        </p>
      )}
      {a.false_positive_check && (
        <p className="ann-fp"><strong>FP check:</strong> {a.false_positive_check}</p>
      )}
      {Object.keys(a.extras).length > 0 && (
        <details className="ann-details">
          <summary>Mechanism &amp; evidence</summary>
          <pre>{JSON.stringify(a.extras, null, 2)}</pre>
        </details>
      )}
    </li>
  );
}
