import type { AgentName } from "../types";
import { ALL_AGENTS, BIAS_COLORS, BIAS_LABELS } from "../types";

interface Props {
  selected: Set<AgentName>;
  setSelected: (s: Set<AgentName>) => void;
}

export function AgentPicker({ selected, setSelected }: Props) {
  const toggle = (name: AgentName) => {
    const next = new Set(selected);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setSelected(next);
  };

  const selectAll = () => setSelected(new Set(ALL_AGENTS.map((a) => a.name)));
  const selectNone = () => setSelected(new Set());

  return (
    <div className="panel">
      <div className="panel-header">
        <p className="overline">02 — Agents</p>
        <h2 className="panel-title">Which detectors to run</h2>
        <p className="hint" style={{ marginTop: 6 }}>
          Each agent is specialised for one bias. Run a single one to focus, or all
          five for full coverage. They run in parallel, so latency is bounded by the
          slowest agent regardless of how many you pick.
        </p>
      </div>

      <div className="row-inline" style={{ marginBottom: "0.75rem" }}>
        <button type="button" className="ghost-btn" onClick={selectAll}>
          Select all
        </button>
        <button type="button" className="ghost-btn" onClick={selectNone}>
          Select none
        </button>
        <span className="hint" style={{ marginLeft: "auto" }}>
          {selected.size} of {ALL_AGENTS.length} selected
        </span>
      </div>

      <div className="agent-grid">
        {ALL_AGENTS.map((a) => {
          const palette = BIAS_COLORS[a.bias_type];
          const isOn = selected.has(a.name);
          return (
            <label
              key={a.name}
              className={`agent-pick ${isOn ? "on" : ""}`}
              style={{
                borderColor: isOn ? palette.fg : "#e2e0d8",
                background: isOn ? palette.bg : "#fff",
              }}
            >
              <input
                type="checkbox"
                checked={isOn}
                onChange={() => toggle(a.name)}
              />
              <div className="agent-pick-body">
                <div className="agent-pick-head">
                  <span
                    className="agent-pick-name"
                    style={{ color: isOn ? palette.fg : "#111" }}
                  >
                    {a.name}
                  </span>
                  <span className="agent-pick-tier">{a.tier}</span>
                </div>
                <div className="agent-pick-bias">{BIAS_LABELS[a.bias_type]}</div>
                <div className="agent-pick-diff">Difficulty · {a.difficulty}</div>
              </div>
            </label>
          );
        })}
      </div>
    </div>
  );
}
