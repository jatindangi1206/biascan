import type { AgentDoneEvent } from "../api";
import type { AgentName, BiasType } from "../types";
import { BIAS_COLORS, BIAS_LABELS } from "../types";

export type AgentStatus =
  | { phase: "waiting" }
  | { phase: "running" }
  | { phase: "done"; raw_count: number; kept_count: number; error: string | null };

interface Props {
  agentNames: AgentName[];
  statuses: Record<string, AgentStatus>;
}

function statusIcon(s: AgentStatus) {
  if (s.phase === "waiting") return <span className="progress-icon waiting" />;
  if (s.phase === "running") return <span className="progress-icon running" />;
  if (s.phase === "done" && s.error) return <span className="progress-icon error">!</span>;
  if (s.phase === "done") return <span className="progress-icon done">✓</span>;
  return null;
}

const AGENT_BIAS: Record<AgentName, BiasType> = {
  ARGUS: "confirmation_bias",
  LIBRA: "certainty_inflation",
  LENS: "overgeneralisation",
  QUILL: "framing_effect",
  VIGIL: "causal_inference_error",
};

// What each agent is responsible for — shown while it is active. These are
// truthful descriptions of the agent's role (not invented per-step progress).
const AGENT_TASK_ACTIVE: Record<AgentName, string> = {
  ARGUS: "Checking for cherry-picked evidence",
  LIBRA: "Reviewing certainty language",
  LENS: "Examining scope and generalisation",
  QUILL: "Evaluating narrative framing",
  VIGIL: "Auditing causal claims",
};

export function ProgressPanel({ agentNames, statuses }: Props) {
  return (
    <section className="progress-panel">
      <p className="section-label">Agent status</p>
      {agentNames.map((name) => {
        const status = statuses[name] ?? { phase: "waiting" };
        const biasType = AGENT_BIAS[name];
        const color = biasType ? BIAS_COLORS[biasType].fg : "#8b8279";
        const activeTask = AGENT_TASK_ACTIVE[name];

        return (
          <div key={name} className={`progress-row phase-${status.phase}`}>
            {statusIcon(status)}
            <span className="progress-agent-dot" style={{ background: color }} />
            <span className="progress-agent-name">{name}</span>
            <span className="progress-agent-label">
              {status.phase === "running"
                ? activeTask
                : biasType
                  ? BIAS_LABELS[biasType]
                  : ""}
            </span>
            <span className="progress-agent-status">
              {status.phase === "waiting" && "Queued"}
              {status.phase === "running" && "Active"}
              {status.phase === "done" && !status.error && (
                status.kept_count === 0
                  ? "No flags"
                  : `${status.kept_count} flag${status.kept_count === 1 ? "" : "s"}`
              )}
              {status.phase === "done" && status.error && "Error"}
            </span>
          </div>
        );
      })}
    </section>
  );
}

export function agentStatusFromEvent(e: AgentDoneEvent): AgentStatus {
  return {
    phase: "done",
    raw_count: e.raw_count,
    kept_count: e.kept_count,
    error: e.error,
  };
}
