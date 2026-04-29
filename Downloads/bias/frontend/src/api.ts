import type {
  AgentInfo,
  AgentName,
  AnalyzeResponse,
  HealthResponse,
  Mode,
  ProviderConfig,
  ProviderInfo,
  Annotation,
  BiasType,
} from "./types";

// ── Types for streaming ────────────────────────────────────────────────────

export interface AgentDoneEvent {
  agent: AgentName;
  bias_type: BiasType;
  prompt_version: string;
  raw_count: number;
  kept_count: number;
  annotations: Annotation[];
  error: string | null;
}

export interface StreamStartEvent {
  document_id: string;
  total_agents: number;
  agent_names: AgentName[];
}

export interface StreamCompletePayload {
  document_id: string;
  overall_bias_score: number;
  annotations: Annotation[];
  agents: Array<{
    agent: string;
    bias_type: BiasType;
    prompt_version: string;
    raw_count: number;
    kept_count: number;
    error: string | null;
  }>;
  mode: Mode;
  warnings: string[];
  provider: { provider: string; model: string; base_url: string | null };
}

// ── Streaming analyze ──────────────────────────────────────────────────────

export function analyzeStream(
  text: string,
  references: string,
  mode: Mode,
  provider: ProviderConfig,
  agents: AgentName[],
  callbacks: {
    onStart: (e: StreamStartEvent) => void;
    onAgentDone: (e: AgentDoneEvent) => void;
    onComplete: (e: StreamCompletePayload) => void;
    onError: (message: string) => void;
  }
): () => void {
  const ctrl = new AbortController();

  (async () => {
    let res: Response;
    try {
      res = await fetch("/api/analyze/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          references: references || null,
          mode,
          provider,
          agents: agents.length === 0 ? null : agents,
        }),
        signal: ctrl.signal,
      });
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") {
        callbacks.onError((e as Error).message ?? String(e));
      }
      return;
    }

    if (!res.ok) {
      callbacks.onError(`HTTP ${res.status}: ${await res.text()}`);
      return;
    }

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE blocks are separated by \n\n
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";

        for (const block of blocks) {
          let eventType = "";
          let dataStr = "";
          for (const line of block.split("\n")) {
            if (line.startsWith("event: ")) eventType = line.slice(7).trim();
            if (line.startsWith("data: ")) dataStr = line.slice(6).trim();
          }
          if (!dataStr) continue;
          try {
            const data = JSON.parse(dataStr);
            if (eventType === "start") callbacks.onStart(data);
            else if (eventType === "agent_done") callbacks.onAgentDone(data);
            else if (eventType === "complete") callbacks.onComplete(data);
            else if (eventType === "error") callbacks.onError(data.message ?? "Unknown error");
          } catch {
            /* ignore malformed SSE data */
          }
        }
      }
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") {
        callbacks.onError((e as Error).message ?? String(e));
      }
    }
  })();

  return () => ctrl.abort();
}

// ── Rest of API ────────────────────────────────────────────────────────────

export async function analyze(
  text: string,
  references: string,
  mode: Mode,
  provider: ProviderConfig,
  agents: AgentName[]
): Promise<AnalyzeResponse> {
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      references: references || null,
      mode,
      provider,
      agents: agents.length === 0 ? null : agents,
    }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json();
}

export async function health(): Promise<HealthResponse> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function listProviders(): Promise<ProviderInfo[]> {
  const res = await fetch("/api/providers");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return data.providers;
}

export async function listAgents(): Promise<AgentInfo[]> {
  const res = await fetch("/api/agents");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return data.agents;
}

export async function pingProvider(
  provider: ProviderConfig
): Promise<{ ok: boolean; sample: string }> {
  const res = await fetch("/api/ping-provider", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider }),
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j?.detail) detail = j.detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}
