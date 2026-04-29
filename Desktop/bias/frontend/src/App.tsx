import { useEffect, useMemo, useRef, useState } from "react";
import { analyzeStream, health, listAgents, listProviders } from "./api";
import type { AgentDoneEvent, StreamCompletePayload } from "./api";
import { InputPanel } from "./components/InputPanel";
import { AnnotatedOutput } from "./components/AnnotatedOutput";
import { ResultsPanel } from "./components/ResultsPanel";
import { ProgressPanel, agentStatusFromEvent } from "./components/ProgressPanel";
import type { AgentStatus } from "./components/ProgressPanel";
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
  HealthResponse,
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
const THEME_KEY     = "biasscan.theme";

const PROVIDER_LABELS = {
  ollama:    "Ollama",
  groq:      "Groq",
  together:  "Together AI",
  nvidia:    "NVIDIA NIM",
  anthropic: "Anthropic",
  openai:    "OpenAI",
  gemini:    "Gemini",
  lightning: "Lightning AI",
} as const;

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
}

const EMPTY_STREAM: StreamState = {
  phase: "idle",
  docId: "",
  agentNames: [],
  agentStatuses: {},
  partialAnnotations: [],
  finalResult: null,
};

export default function App() {
  const [text, setText] = useState("");
  const [references, setReferences] = useState("");
  const [mode, setMode] = useState<Mode>("lite");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [entered, setEntered] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [healthInfo, setHealthInfo] = useState<HealthResponse | null>(null);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [bootWarning, setBootWarning] = useState<string | null>(null);
  const [stream, setStream] = useState<StreamState>(EMPTY_STREAM);

  const abortRef = useRef<(() => void) | null>(null);

  // ── Theme ───────────────────────────────────────────────────────────────
  const [isDark, setIsDark] = useState<boolean>(() => {
    try {
      const stored = localStorage.getItem(THEME_KEY);
      if (stored !== null) return stored === "dark";
    } catch { /* ignore */ }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  });

  useEffect(() => {
    const root = document.documentElement;
    if (isDark) {
      root.setAttribute("data-theme", "dark");
    } else {
      root.removeAttribute("data-theme");
    }
    try { localStorage.setItem(THEME_KEY, isDark ? "dark" : "light"); } catch { /* ignore */ }
  }, [isDark]);
  // ────────────────────────────────────────────────────────────────────────

  const [config, setConfig] = useState<ProviderConfig>(
    () => loadStoredConfig() ?? DEFAULT_CONFIG
  );
  const [selected, setSelected] = useState<Set<AgentName>>(
    () => new Set(loadStoredAgents())
  );

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([health(), listProviders(), listAgents()]).then(
      ([healthResult, providersResult, agentsResult]) => {
        if (cancelled) return;
        const warnings: string[] = [];

        if (healthResult.status === "fulfilled") {
          setHealthInfo(healthResult.value);
        } else {
          warnings.push("Backend health could not be loaded.");
        }

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
  const providerLabel =
    providers.find((p) => p.name === config.provider)?.label.split(" (")[0] ??
    PROVIDER_LABELS[config.provider];
  const promptLabel = healthInfo ? `Prompt ${healthInfo.prompt_version}` : "Prompt unknown";

  const onAnalyze = () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setEntered(true);
    setSettingsOpen(false);

    // Initialise stream state — all agents shown as "waiting"
    const agentList = Array.from(selected);
    const initialStatuses: Record<string, AgentStatus> = {};
    for (const n of agentList) initialStatuses[n] = { phase: "waiting" };

    setStream({
      phase: "streaming",
      docId: "",
      agentNames: agentList,
      agentStatuses: initialStatuses,
      partialAnnotations: [],
      finalResult: null,
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
            // Mark all as "running" once the server confirms it started
            const statuses: Record<string, AgentStatus> = {};
            for (const n of e.agent_names) statuses[n] = { phase: "running" };
            return { ...prev, docId: e.document_id, agentNames: e.agent_names, agentStatuses: statuses };
          });
        },
        onAgentDone(e: AgentDoneEvent) {
          setStream((prev) => ({
            ...prev,
            agentStatuses: {
              ...prev.agentStatuses,
              [e.agent]: agentStatusFromEvent(e),
            },
            partialAnnotations: [...prev.partialAnnotations, ...e.annotations],
          }));
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
          setStream((prev) => ({ ...prev, phase: "done", finalResult }));
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
  };

  // Decide what to show in the analysis column
  const isStreaming = stream.phase === "streaming";
  const displayAnnotations =
    result?.annotations ?? stream.partialAnnotations;
  const displayResult = result;

  return (
    <div className={`app-shell stage-${stage}`}>
      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        healthInfo={healthInfo}
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
                disabled={loading || !canAnalyze}
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
          {stage === "landing" ? (
            <span className="preview-tag">
              {healthInfo ? `${promptLabel} · research preview` : "Research preview"}
            </span>
          ) : (
            <button
              type="button"
              className="nav-link"
              onClick={() => setSettingsOpen(true)}
            >
              Settings
            </button>
          )}
          <button
            type="button"
            className="theme-toggle"
            title={isDark ? "Switch to light mode" : "Switch to dark mode"}
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
            onClick={() => setIsDark((d) => !d)}
          >
            {isDark ? "☀" : "☾"}
          </button>
        </div>
      </header>

      {bootWarning && <div className="boot-banner">{bootWarning}</div>}
      {error && <div className="error-banner">{error}</div>}

      <main className="main-stage">
        {stage === "landing" && (
          <section className="landing-stage">
            <p className="eyebrow">A focused thinking tool</p>
            <h1 className="landing-title">BiasScan</h1>
            <p className="landing-copy">
              Detect cognitive bias in scientific synthesis — one paragraph at a time.
            </p>
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
              providerLabel={providerLabel}
              modelLabel={config.model}
              backendLabel={promptLabel}
              agentCount={selected.size}
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
                <p className="section-label">Original</p>
                <div className="reading-surface">{text}</div>
              </article>

              <article className="analysis-column">
                {/* Live agent progress — shown while streaming */}
                {isStreaming && (
                  <ProgressPanel
                    agentNames={stream.agentNames}
                    statuses={stream.agentStatuses}
                  />
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
      </main>

      <footer className="app-footer">
        <span>
          Five agents · Confirmation · Certainty · Overgeneralisation · Framing ·
          Causal
        </span>
        <span className="footer-line" />
      </footer>
    </div>
  );
}
