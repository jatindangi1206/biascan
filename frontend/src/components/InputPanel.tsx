import { useState } from "react";
import type { Mode } from "../types";

const MODE_META: Record<
  Mode,
  { label: string; description: string; tooltip: string; available: boolean }
> = {
  lite: {
    label: "Lite",
    description: "Text-only scan. Fastest.",
    tooltip: "Reads only the text you paste. No outside lookups.",
    available: true,
  },
  adaptive: {
    label: "Adaptive",
    description: "Pulls a summary of cited studies for cross-checking.",
    tooltip:
      "When a flag says 'RAG check needed', BiasScan pulls a summary of the study you cited and lets the agent re-check the claim against the source.",
    available: false,
  },
  premium: {
    label: "Premium",
    description: "Fetches the full cited paper and audits the source itself.",
    tooltip:
      "When the cited study is open access, BiasScan retrieves the full text, runs bias detection on the source's own claims, and uses citation counts as a quality signal.",
    available: false,
  },
};

interface Props {
  text: string;
  setText: (s: string) => void;
  references: string;
  setReferences: (s: string) => void;
  mode: Mode;
  setMode: (m: Mode) => void;
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
  wordCap,
  canAnalyze,
  loading,
  onAnalyze,
}: Props) {
  const [showReferences, setShowReferences] = useState(false);
  const [showAdvancedModes, setShowAdvancedModes] = useState(false);
  const wordCount = countWords(text);
  const overCap = wordCount > wordCap;

  return (
    <div className="composer">
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
        {overCap && (
          <p className="input-cap-note">
            Over the {wordCap.toLocaleString()}-word cap — only the first {wordCap.toLocaleString()} words will be analysed.
          </p>
        )}
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
          <button
            type="button"
            className={`mode-tab ${mode === "lite" ? "active" : ""}`}
            onClick={() => setMode("lite")}
          >
            {MODE_META.lite.label}
          </button>

          {showAdvancedModes &&
            (["adaptive", "premium"] as Mode[]).map((m) => (
              <span
                key={m}
                className="mode-tab disabled"
                data-tooltip={MODE_META[m].tooltip}
                aria-disabled="true"
              >
                {MODE_META[m].label}
                <span className="soon-tag">soon</span>
              </span>
            ))}

          <button
            type="button"
            className="mode-expand"
            onClick={() => setShowAdvancedModes((v) => !v)}
            aria-expanded={showAdvancedModes}
          >
            {showAdvancedModes ? "Less" : "More modes"}
            <span className="mode-chevron">{showAdvancedModes ? "▴" : "▾"}</span>
          </button>
        </div>
        <p className="mode-description">
          {MODE_META[mode].description}
          {showAdvancedModes && (
            <span className="mode-soon-note"> · Adaptive and Premium are rolling out soon.</span>
          )}
        </p>
      </div>

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
