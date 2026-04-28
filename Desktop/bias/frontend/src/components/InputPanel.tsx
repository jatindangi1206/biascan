import type { Mode } from "../types";

interface Props {
  text: string;
  setText: (s: string) => void;
  references: string;
  setReferences: (s: string) => void;
  mode: Mode;
  setMode: (m: Mode) => void;
  onAnalyze: () => void;
  loading: boolean;
}

const SAMPLE = `These findings clearly demonstrate that intervention X causes a substantial reduction in anxiety symptoms across all patient populations. The accumulating evidence definitively confirms our hypothesis, with three randomised trials showing significant benefits. While one observational study reported null results, methodological limitations preclude drawing strong conclusions from that work. The intervention reduces risk of relapse by 40%, offering patients a meaningful chance of recovery. Increased adherence leads to improved long-term outcomes — the mechanism is clear.`;

export function InputPanel({
  text,
  setText,
  references,
  setReferences,
  mode,
  setMode,
  onAnalyze,
  loading,
}: Props) {
  return (
    <div className="panel">
      <div className="panel-header">
        <p className="overline">01 — Input</p>
        <h2 className="panel-title">Synthesis text + references</h2>
      </div>

      <label className="label">Synthesis / results section</label>
      <textarea
        className="textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste the results or synthesis section of the systematic review here..."
        rows={14}
      />
      <div className="row-between">
        <span className="hint">{text.length.toLocaleString()} characters</span>
        <button
          type="button"
          className="ghost-btn"
          onClick={() => setText(SAMPLE)}
          disabled={loading}
        >
          Load sample
        </button>
      </div>

      <label className="label" style={{ marginTop: "1.25rem" }}>
        Reference list (optional)
      </label>
      <textarea
        className="textarea"
        value={references}
        onChange={(e) => setReferences(e.target.value)}
        placeholder="Paste numbered references or DOIs, one per line. Used by Premium mode."
        rows={5}
      />

      <div className="mode-row">
        <span className="label" style={{ marginRight: 12 }}>Mode</span>
        {(["lite", "adaptive", "premium"] as const).map((m) => (
          <button
            key={m}
            type="button"
            className={`mode-pill ${mode === m ? "active" : ""}`}
            onClick={() => setMode(m)}
            disabled={loading}
          >
            {m === "lite" && "Lite · text only"}
            {m === "adaptive" && "Adaptive"}
            {m === "premium" && "Premium · RAG"}
          </button>
        ))}
      </div>

      <button
        type="button"
        className="primary-btn"
        onClick={onAnalyze}
        disabled={loading || text.trim().length === 0}
      >
        {loading ? "Running 5 agents in parallel…" : "Analyse for bias"}
      </button>
    </div>
  );
}
