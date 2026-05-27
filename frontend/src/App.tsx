import { useEffect, useMemo, useRef, useState } from "react";
import { analyzeStream, listAgents, listProviders } from "./api";
import type {
  AgentDoneEvent,
  StreamCompletePayload,
  StreamPipelineMetaEvent,
  StreamAgentStartedEvent,
  StreamAegisStartedEvent,
  StreamAegisDoneEvent,
} from "./api";
import { InputPanel } from "./components/InputPanel";
import { AnnotatedOutput } from "./components/AnnotatedOutput";
import { ResultsPanel } from "./components/ResultsPanel";
import { ProgressPanel, agentStatusFromEvent } from "./components/ProgressPanel";
import type { AgentStatus } from "./components/ProgressPanel";
import { AnalysisLog, nowStamp } from "./components/AnalysisLog";
import type { LogEntry } from "./components/AnalysisLog";
import { HowItWorks } from "./components/HowItWorks";
import {
  SettingsPanel,
  loadStoredConfig,
  storeConfig,
} from "./components/SettingsPanel";
import type {
  AgentInfo,
  AgentName,
  AnalyzeResponse,
  Annotation,
  Mode,
  ProviderConfig,
  ProviderInfo,
} from "./types";
import { DEFAULT_AGENT_NAMES } from "./types";

const DEFAULT_CONFIG: ProviderConfig = {
  provider: "ollama",
  model: "qwen2.5:7b",
  api_key: "",
  base_url: "http://localhost:11434",
};

const SELECTION_KEY = "biasscan.agents";
const EVALUATION_NOTICE =
  "Currently under active evaluation. Results may be experimental, or subject to change.";

function loadStoredAgents(): AgentName[] {
  try {
    const raw = localStorage.getItem(SELECTION_KEY);
    if (!raw) return DEFAULT_AGENT_NAMES;
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed as AgentName[];
  } catch { /* ignore */ }
  return DEFAULT_AGENT_NAMES;
}

// ── Stream state ────────────────────────────────────────────────────────────

type StreamPhase = "idle" | "streaming" | "done";

interface StreamState {
  phase: StreamPhase;
  docId: string;
  agentNames: AgentName[];
  agentStatuses: Record<string, AgentStatus>;
  partialAnnotations: Annotation[];
  finalResult: AnalyzeResponse | null;
  logEntries: LogEntry[];
}

const EMPTY_STREAM: StreamState = {
  phase: "idle",
  docId: "",
  agentNames: [],
  agentStatuses: {},
  partialAnnotations: [],
  finalResult: null,
  logEntries: [],
};

function appendLog(prev: LogEntry[], text: string, kind: LogEntry["kind"] = "info"): LogEntry[] {
  return [...prev, { ts: nowStamp(), text, kind }];
}

export default function App() {
  const [text, setText] = useState("");
  const [references, setReferences] = useState("");
  const [mode, setMode] = useState<Mode>("lite");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [entered, setEntered] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [showHowItWorks, setShowHowItWorks] = useState(false);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [bootWarning, setBootWarning] = useState<string | null>(null);
  const [stream, setStream] = useState<StreamState>(EMPTY_STREAM);
  const [lastAnalyzedText, setLastAnalyzedText] = useState("");
  const [isEditingOriginal, setIsEditingOriginal] = useState(false);

  const abortRef = useRef<(() => void) | null>(null);

  const [config, setConfig] = useState<ProviderConfig>(
    () => loadStoredConfig() ?? DEFAULT_CONFIG
  );
  const [selected, setSelected] = useState<Set<AgentName>>(
    () => new Set(loadStoredAgents())
  );

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([listProviders(), listAgents()]).then(
      ([providersResult, agentsResult]) => {
        if (cancelled) return;
        const warnings: string[] = [];

        if (providersResult.status === "fulfilled") {
          const nextProviders = providersResult.value;
          setProviders(nextProviders);
          if (!loadStoredConfig()) {
            const preferred =
              nextProviders.find((p) => p.name === DEFAULT_CONFIG.provider) ??
              nextProviders[0];
            if (preferred) {
              const nextConfig: ProviderConfig = {
                provider: preferred.name,
                model: preferred.default_model,
                api_key: "",
                base_url: preferred.default_base_url,
              };
              setConfig(nextConfig);
              storeConfig(nextConfig);
            }
          }
        } else {
          warnings.push("Provider metadata could not be loaded.");
        }

        if (agentsResult.status === "fulfilled") {
          setAgents(agentsResult.value);
        } else {
          warnings.push("Agent metadata could not be loaded.");
        }

        setBootWarning(warnings.length > 0 ? warnings.join(" ") : null);
      }
    );
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    localStorage.setItem(SELECTION_KEY, JSON.stringify(Array.from(selected)));
  }, [selected]);

  useEffect(() => {
    if (agents.length === 0) return;
    const validNames = new Set(agents.map((a) => a.name));
    setSelected((current) => {
      const filtered = Array.from(current).filter((n) => validNames.has(n));
      const next =
        filtered.length > 0
          ? new Set(filtered)
          : new Set(agents.map((a) => a.name));
      if (next.size === current.size && Array.from(next).every((n) => current.has(n)))
        return current;
      return next;
    });
  }, [agents]);

  const canAnalyze = useMemo(() => {
    if (!text.trim()) return false;
    if (selected.size === 0) return false;
    if (!config.model) return false;
    if (config.provider !== "ollama" && !config.api_key) return false;
    return true;
  }, [text, selected, config]);

  const stage = result || stream.phase !== "idle" ? "result" : entered ? "compose" : "landing";
  const currentProvider = providers.find((p) => p.name === config.provider);
  const wordCap = currentProvider?.word_cap ?? 8000;

  const onAnalyze = () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setEntered(true);
    setSettingsOpen(false);
    setLastAnalyzedText(text);

    // Initialise stream state — all agents shown as "waiting"
    const agentList = Array.from(selected);
    const initialStatuses: Record<string, AgentStatus> = {};
    for (const n of agentList) initialStatuses[n] = { phase: "waiting" };

    const charCount = text.length.toLocaleString();
    setStream({
      phase: "streaming",
      docId: "",
      agentNames: agentList,
      agentStatuses: initialStatuses,
      partialAnnotations: [],
      finalResult: null,
      logEntries: appendLog([], `Sent ${charCount} chars to BiasScan`, "info"),
    });

    const abort = analyzeStream(
      text,
      references,
      mode,
      config,
      agentList,
      {
        onStart(e) {
          setStream((prev) => {
            // Mark all as "running" once the server confirms it started.
            // Per-agent activation copy is appended on each agent_started
            // event below.
            const statuses: Record<string, AgentStatus> = {};
            for (const n of e.agent_names) statuses[n] = { phase: "running" };
            return {
              ...prev,
              docId: e.document_id,
              agentNames: e.agent_names,
              agentStatuses: statuses,
              logEntries: appendLog(
                prev.logEntries,
                `Pipeline started — ${e.agent_names.length} agents queued`,
                "info"
              ),
            };
          });
        },
        onPipelineMeta(e: StreamPipelineMetaEvent) {
          setStream((prev) => {
            // The pipeline_meta event carries the backend's initial warnings
            // (e.g. "Input RAG active: document chunked into 5 segments.")
            // which already encode the chunk count. Logging them is enough;
            // we don't separately surface e.chunks here.
            let log = prev.logEntries;
            for (const w of e.warnings) log = appendLog(log, w, "info");
            return { ...prev, logEntries: log };
          });
        },
        onAgentStarted(e: StreamAgentStartedEvent) {
          setStream((prev) => ({
            ...prev,
            logEntries: appendLog(prev.logEntries, `${e.agent} scanning…`, "agent"),
          }));
        },
        onAegisStarted(e: StreamAegisStartedEvent) {
          setStream((prev) => ({
            ...prev,
            logEntries: appendLog(
              prev.logEntries,
              `AEGIS resolving ${e.conflicts} conflict${e.conflicts === 1 ? "" : "s"}…`,
              "agent"
            ),
          }));
        },
        onAegisDone(e: StreamAegisDoneEvent) {
          setStream((prev) => ({
            ...prev,
            logEntries: appendLog(
              prev.logEntries,
              `AEGIS resolved ${e.resolved} conflict${e.resolved === 1 ? "" : "s"}`,
              "done"
            ),
          }));
        },
        onAgentDone(e: AgentDoneEvent) {
          setStream((prev) => {
            const summary = e.error
              ? `${e.agent} error — ${e.error}`
              : e.kept_count === 0
                ? `${e.agent} complete — no flags`
                : `${e.agent} complete — ${e.kept_count} flag${e.kept_count === 1 ? "" : "s"}`;
            return {
              ...prev,
              agentStatuses: {
                ...prev.agentStatuses,
                [e.agent]: agentStatusFromEvent(e),
              },
              partialAnnotations: [...prev.partialAnnotations, ...e.annotations],
              logEntries: appendLog(
                prev.logEntries,
                summary,
                e.error ? "error" : e.kept_count > 0 ? "flag" : "done"
              ),
            };
          });
        },
        onComplete(e: StreamCompletePayload) {
          const finalResult: AnalyzeResponse = {
            document_id: e.document_id,
            mode: e.mode as Mode,
            overall_bias_score: e.overall_bias_score,
            annotations: e.annotations,
            agents: e.agents.map((a) => ({
              agent: a.agent as AgentName,
              bias_type: a.bias_type,
              prompt_version: a.prompt_version,
              raw_count: a.raw_count,
              kept_count: a.kept_count,
              error: a.error,
            })),
            warnings: e.warnings,
            provider: e.provider as AnalyzeResponse["provider"],
          };
          setStream((prev) => {
            // Initial warnings were already logged via pipeline_meta; any
            // late-added warnings (e.g. agent errors) are already surfaced
            // via their agent_done error field, so we don't re-iterate
            // e.warnings here.
            const log = appendLog(
              prev.logEntries,
              `Analysis complete — score ${e.overall_bias_score.toFixed(2)}`,
              "done"
            );
            return { ...prev, phase: "done", finalResult, logEntries: log };
          });
          setResult(finalResult);
          setLoading(false);
        },
        onError(message: string) {
          setError(message);
          setStream(EMPTY_STREAM);
          setLoading(false);
        },
      }
    );

    abortRef.current = abort;
  };

  const startCompose = () => {
    setEntered(true);
    setResult(null);
    setError(null);
    setStream(EMPTY_STREAM);
  };

  const startNewScan = () => {
    if (abortRef.current) { abortRef.current(); abortRef.current = null; }
    setEntered(true);
    setResult(null);
    setError(null);
    setLoading(false);
    setStream(EMPTY_STREAM);
    setText("");
    setReferences("");
    setLastAnalyzedText("");
    setIsEditingOriginal(false);
  };

  // Decide what to show in the analysis column
  const isStreaming = stream.phase === "streaming";
  const displayAnnotations =
    result?.annotations ?? stream.partialAnnotations;
  const displayResult = result;
  const hasEditedText = text.trim() !== lastAnalyzedText.trim();
  const canRunAgain = canAnalyze && hasEditedText && !loading;

  return (
    <div className={`app-shell stage-${stage}`}>
      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        providers={providers}
        agents={agents}
        config={config}
        setConfig={setConfig}
        selected={selected}
        setSelected={setSelected}
      />

      {settingsOpen && (
        <button
          type="button"
          className="settings-backdrop"
          aria-label="Close settings"
          onClick={() => setSettingsOpen(false)}
        />
      )}

      <header className="topbar">
        <div className="topbar-left">
          {stage === "result" && (
            <>
              <button type="button" className="nav-link" onClick={startNewScan}>
                New scan
              </button>
              <button
                type="button"
                className="nav-link nav-link-primary"
                onClick={onAnalyze}
                disabled={!canRunAgain}
              >
                {loading ? "Running…" : "Run again"}
              </button>
            </>
          )}
          <div className="brand-mark">
            <span className="brand-dot" />
            <span className="brand-name">BiasScan</span>
          </div>
        </div>

          <div className="topbar-right">
            {!showHowItWorks && stage !== "landing" && (
              <button
                type="button"
                className="nav-link"
                onClick={() => setSettingsOpen(true)}
              >
                Settings
              </button>
            )}
        </div>
      </header>

      <div className="research-banner">{EVALUATION_NOTICE}</div>
      {bootWarning && !showHowItWorks && <div className="boot-banner">{bootWarning}</div>}
      {error && !showHowItWorks && <div className="error-banner">{error}</div>}

      {showHowItWorks && (
        <main className="main-stage">
          <HowItWorks onBack={() => setShowHowItWorks(false)} />
        </main>
      )}

      {!showHowItWorks && <main className="main-stage">
        {stage === "landing" && (
          <section className="landing-stage">
            <p className="eyebrow">A focused thinking tool</p>
            <h1 className="landing-title">BiasScan</h1>
            <p className="landing-copy">Detect cognitive bias in scientific synthesis</p>
            <button type="button" className="hero-button" onClick={startCompose}>
              Try Now
            </button>
          </section>
        )}

        {stage === "compose" && (
          <section className="compose-stage">
            <InputPanel
              text={text}
              setText={setText}
              references={references}
              setReferences={setReferences}
              mode={mode}
              setMode={setMode}
              wordCap={wordCap}
              canAnalyze={canAnalyze}
              loading={loading}
              onAnalyze={onAnalyze}
            />
          </section>
        )}

        {stage === "result" && (
          <section className="result-stage">
            <div className="result-grid">
              <article className="reading-column">
                <div className="reading-head">
                  <p className="section-label">Original</p>
                  <button
                    type="button"
                    className="mini-action"
                    onClick={() => setIsEditingOriginal((prev) => !prev)}
                  >
                    {isEditingOriginal ? "Done" : "Edit text"}
                  </button>
                </div>
                {isEditingOriginal ? (
                  <textarea
                    className="reading-editor"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    rows={16}
                  />
                ) : (
                  <div className="reading-surface">{text}</div>
                )}
              </article>

              <article className="analysis-column">
                {/* Live agent progress — shown while streaming */}
                {isStreaming && (
                  <>
                    <ProgressPanel
                      agentNames={stream.agentNames}
                      statuses={stream.agentStatuses}
                    />
                    <AnalysisLog entries={stream.logEntries} />
                  </>
                )}

                {/* Results panel — shown once complete */}
                {displayResult && <ResultsPanel result={displayResult} />}

                {/* Annotations — accumulate live during streaming */}
                {displayAnnotations.length > 0 && (
                  <AnnotatedOutput
                    text={text}
                    annotations={displayAnnotations}
                  />
                )}

                {/* Empty state while first agent is working */}
                {isStreaming && displayAnnotations.length === 0 && (
                  <p className="stream-empty-hint">
                    Agents are scanning your text…
                  </p>
                )}
              </article>
            </div>
          </section>
        )}
      </main>}

      <footer className="app-footer">
        <span className="footer-preview">Research Preview</span>
        <button
          type="button"
          className="footer-link"
          onClick={() => setShowHowItWorks(true)}
        >
          How it works
        </button>
      </footer>
    </div>
  );
}
