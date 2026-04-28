import type { AnalyzeResponse, Mode } from "./types";

export async function analyze(
  text: string,
  references: string,
  mode: Mode
): Promise<AnalyzeResponse> {
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, references: references || null, mode }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json();
}

export interface HealthResponse {
  status: string;
  model: string;
  prompt_version: string;
  anthropic_key_set: boolean;
}

export async function health(): Promise<HealthResponse> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
