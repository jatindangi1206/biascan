import { useEffect, useState } from "react";
import { analyze, health } from "./api";
import type { AnalyzeResponse, Mode } from "./types";
import { InputPanel } from "./components/InputPanel";
import { AnnotatedOutput } from "./components/AnnotatedOutput";
import { ResultsPanel } from "./components/ResultsPanel";

export default function App() {
  const [text, setText] = useState("");
  const [references, setReferences] = useState("");
  const [mode, setMode] = useState<Mode>("lite");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [keySet, setKeySet] = useState<boolean | null>(null);
  const [model, setModel] = useState<string>("");

  useEffect(() => {
    health()
      .then((h) => {
        setKeySet(h.anthropic_key_set);
        setModel(h.model);
      })
      .catch(() => setKeySet(false));
  }, []);

  const onAnalyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await analyze(text, references, mode);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="wrap">
      <header className="title-block">
        <p className="overtitle">Trivedi School of Biosciences · Ashoka University</p>
        <h1 className="main-title">BiasScan</h1>
        <p className="subtitle-doc">
          Agentic cognitive bias detection for AI-generated systematic review synthesis text.
          Five specialised agents — ARGUS, LIBRA, LENS, QUILL, VIGIL — run in parallel.
        </p>
        <div className="status-row">
          <span className={`dot ${keySet ? "dot-ok" : "dot-warn"}`} />
          <span className="status-text">
            {keySet === null
              ? "checking server…"
              : keySet
                ? `Live · ${model}`
                : "API key not set on server — agents will return empty results"}
          </span>
        </div>
      </header>

      <InputPanel
        text={text}
        setText={setText}
        references={references}
        setReferences={setReferences}
        mode={mode}
        setMode={setMode}
        onAnalyze={onAnalyze}
        loading={loading}
      />

      {error && <div className="error-box">Error: {error}</div>}

      {result && (
        <>
          <AnnotatedOutput text={text} annotations={result.annotations} />
          <ResultsPanel result={result} />
        </>
      )}

      <footer className="footer">
        <p>v0.1 · prompts versioned at <code>prompts/v1/*</code></p>
      </footer>
    </div>
  );
}
