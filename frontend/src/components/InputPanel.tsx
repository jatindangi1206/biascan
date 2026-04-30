import { useState } from "react";
import type { Mode } from "../types";

const MODE_META: Record<Mode, { label: string; description: string }> = {
  lite:     { label: "Lite",     description: "Text-only scan, fastest." },
  adaptive: { label: "Adaptive", description: "Lite first; escalates if bias found." },
  premium:  { label: "Premium",  description: "Reference-aware — checks your citations." },
};

interface Props {
  text: string;
  setText: (s: string) => void;
  references: string;
  setReferences: (s: string) => void;
  mode: Mode;
  setMode: (m: Mode) => void;
  providerLabel: string;
  modelLabel: string;
  backendLabel: string;
  agentCount: number;
  wordCap: number;
  canAnalyze: boolean;
  loading: boolean;
  onAnalyze: () => void;
}

function countWords(s: string): number {
  const trimmed = s.trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).length;
}

const SAMPLE = `These findings clearly demonstrate that intervention X causes a substantial reduction in anxiety symptoms across all patient populations. The accumulating evidence definitively confirms our hypothesis, with three randomised trials showing significant benefits. While one observational study reported null results, methodological limitations preclude drawing strong conclusions from that work. The intervention reduces risk of relapse by 40%, offering patients a meaningful chance of recovery. Increased adherence leads to improved long-term outcomes — the mechanism is clear.`;

export function InputPanel({
  text,
  setText,
  references,
  setReferences,
  mode,
  setMode,
  providerLabel,
  modelLabel,
  backendLabel,
  agentCount,
  wordCap,
  canAnalyze,
  loading,
  onAnalyze,
}: Props) {
  const [showReferences, setShowReferences] = useState(false);
  const wordCount = countWords(text);
  const overCap = wordCount > wordCap;

  return (
    <div className="composer">
      <p className="eyebrow">Paste a results or synthesis section</p>
      <h2 className="composer-title">What would you like to scan?</h2>

      <div className="input-shell">
        <textarea
          className="composer-textarea"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste your synthesis text here..."
          rows={10}
        />

        <div className="input-meta">
          <button
            type="button"
            className="inline-action"
            onClick={() => setText(SAMPLE)}
          >
            Load sample
          </button>
          <span>{text.length.toLocaleString()} chars</span>
          <span className={overCap ? "input-words over-limit" : "input-words"}>
            {wordCount.toLocaleString()} / {wordCap.toLocaleString()} words
          </span>
        </div>
        <p className="input-cap-note">
          {overCap
            ? `Over the ${providerLabel} cap — only the first ${wordCap.toLocaleString()} words will be analysed.`
            : `${providerLabel} cap: ${wordCap.toLocaleString()} words. Capped for cost; raise word_cap in providers/base.py if your API limits allow.`}
        </p>
      </div>

      <div className="references-block">
        <button
          type="button"
          className={`collapse-toggle ${showReferences ? "open" : ""}`}
          onClick={() => setShowReferences((v) => !v)}
        >
          References (optional)
        </button>

        {showReferences && (
          <textarea
            className="references-textarea"
            value={references}
            onChange={(e) => setReferences(e.target.value)}
            placeholder="Paste numbered references or DOIs, one per line."
            rows={5}
          />
        )}
      </div>

      <div className="mode-picker">
        <div className="mode-tabs">
          {(["lite", "adaptive", "premium"] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              className={`mode-tab ${mode === m ? "active" : ""}`}
              onClick={() => setMode(m)}
            >
              {MODE_META[m].label}
            </button>
          ))}
        </div>
        <p className="mode-description">{MODE_META[mode].description}</p>
      </div>

      <p className="composer-status">
        {providerLabel} · {modelLabel} · {agentCount} agent
        {agentCount === 1 ? "" : "s"} · {backendLabel}
      </p>

      <button
        type="button"
        className="run-button"
        disabled={loading || !canAnalyze}
        onClick={onAnalyze}
      >
        {loading ? "Running analysis..." : "Run Analysis"}
      </button>
    </div>
  );
}
