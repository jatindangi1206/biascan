import { useState } from "react";
import { pingProvider } from "../api";
import { BIAS_LABELS } from "../types";
import type {
  AgentInfo,
  AgentName,
  HealthResponse,
  ProviderConfig,
  ProviderInfo,
  ProviderName,
} from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
  healthInfo: HealthResponse | null;
  providers: ProviderInfo[];
  agents: AgentInfo[];
  config: ProviderConfig;
  setConfig: (c: ProviderConfig) => void;
  selected: Set<AgentName>;
  setSelected: (s: Set<AgentName>) => void;
}

const STORAGE_KEY = "biasscan.provider";

export function loadStoredConfig(): ProviderConfig | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as ProviderConfig;
  } catch {
    return null;
  }
}

export function storeConfig(c: ProviderConfig) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(c));
}

export function SettingsPanel({
  open,
  onClose,
  healthInfo,
  providers,
  agents,
  config,
  setConfig,
  selected,
  setSelected,
}: Props) {
  const [showKey, setShowKey] = useState(false);
  const [pingState, setPingState] = useState<
    | { kind: "idle" }
    | { kind: "running" }
    | { kind: "ok"; sample: string }
    | { kind: "err"; message: string }
  >({ kind: "idle" });

  const current = providers.find((provider) => provider.name === config.provider);

  const update = (patch: Partial<ProviderConfig>) => {
    const next = { ...config, ...patch };
    setConfig(next);
    storeConfig(next);
    setPingState({ kind: "idle" });
  };

  const switchProvider = (name: ProviderName) => {
    const info = providers.find((provider) => provider.name === name);
    const next: ProviderConfig = info
      ? {
          provider: name,
          model: info.default_model,
          api_key: "",
          base_url: info.default_base_url,
        }
      : { provider: name, model: "", api_key: "", base_url: null };
    setConfig(next);
    storeConfig(next);
    setPingState({ kind: "idle" });
  };

  const toggleAgent = (name: AgentName) => {
    const next = new Set(selected);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setSelected(next);
  };

  const handlePing = async () => {
    setPingState({ kind: "running" });
    try {
      const response = await pingProvider(config);
      setPingState({ kind: "ok", sample: response.sample });
    } catch (e) {
      setPingState({
        kind: "err",
        message: e instanceof Error ? e.message : String(e),
      });
    }
  };

  return (
    <aside className={`settings-drawer ${open ? "open" : ""}`}>
      <div className="drawer-head">
        <p className="drawer-kicker">Settings</p>
        <button type="button" className="drawer-close" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="drawer-body">
        <section className="drawer-section">
          <p className="drawer-label">Backend</p>
          <div className="drawer-meta-card">
            <p>
              Status · <strong>{healthInfo?.status ?? "unknown"}</strong>
            </p>
            <p>
              Prompt version · <strong>{healthInfo?.prompt_version ?? "unknown"}</strong>
            </p>
            {healthInfo?.key_storage && <p>{healthInfo.key_storage}</p>}
          </div>
        </section>

        <section className="drawer-section">
          <p className="drawer-label">Provider</p>
          <select
            className="drawer-select"
            value={config.provider}
            onChange={(e) => switchProvider(e.target.value as ProviderName)}
          >
            {providers.map((provider) => (
              <option key={provider.name} value={provider.name}>
                {provider.label}
              </option>
            ))}
          </select>
        </section>

        <section className="drawer-section">
          <p className="drawer-label">Model and endpoint</p>
          <input
            className="drawer-input"
            value={config.model}
            onChange={(e) => update({ model: e.target.value })}
            placeholder={current?.default_model ?? ""}
          />
          {current?.model_hint && <p className="drawer-note">{current.model_hint}</p>}
          <input
            className="drawer-input"
            value={config.base_url ?? ""}
            onChange={(e) => update({ base_url: e.target.value })}
            placeholder={current?.default_base_url ?? "Base URL"}
          />
          <p className="drawer-note">
            {current?.needs_base_url
              ? "Required for this provider."
              : "Optional override. Useful for proxies, gateways, or self-hosted compatible endpoints."}
          </p>

          {current?.needs_key && (
            <div className="key-row">
              <input
                className="drawer-input key-input"
                type={showKey ? "text" : "password"}
                autoComplete="off"
                value={config.api_key ?? ""}
                onChange={(e) => update({ api_key: e.target.value })}
                placeholder="API key"
              />
              <button
                type="button"
                className="mini-action"
                onClick={() => setShowKey((value) => !value)}
              >
                {showKey ? "Hide" : "Show"}
              </button>
            </div>
          )}

          <div className="test-row">
            <button
              type="button"
              className="mini-action"
              onClick={handlePing}
              disabled={
                pingState.kind === "running" ||
                !config.model ||
                (current?.needs_key && !config.api_key)
              }
            >
              {pingState.kind === "running" ? "Testing..." : "Test connection"}
            </button>
            {pingState.kind === "ok" && (
              <span className="status-copy">Reachable · {pingState.sample}</span>
            )}
            {pingState.kind === "err" && (
              <span className="status-copy error">{pingState.message}</span>
            )}
          </div>
        </section>

        <section className="drawer-section">
          <div className="drawer-label-row">
            <p className="drawer-label">Agents</p>
            <div className="drawer-inline-actions">
              <button
                type="button"
                className="mini-action"
                onClick={() => setSelected(new Set(agents.map((agent) => agent.name)))}
              >
                All
              </button>
              <button
                type="button"
                className="mini-action"
                onClick={() => setSelected(new Set())}
              >
                None
              </button>
            </div>
          </div>

          <div className="agent-stack">
            {agents.map((agent) => {
              const isOn = selected.has(agent.name);
              return (
                <button
                  key={agent.name}
                  type="button"
                  className={`agent-option ${isOn ? "active" : ""}`}
                  onClick={() => toggleAgent(agent.name)}
                >
                  <span className="agent-bullet" />
                  <span className="agent-copy">
                    <strong>{BIAS_LABELS[agent.bias_type]}</strong>
                    <small>
                      {agent.name} · {agent.prompt_version}
                    </small>
                  </span>
                  <span className="agent-state">{isOn ? "On" : "Off"}</span>
                </button>
              );
            })}
          </div>
        </section>
      </div>
    </aside>
  );
}
