export type Mode = "lite" | "premium" | "adaptive";
export type Severity = "low" | "medium" | "high";
export type BiasType =
  | "confirmation_bias"
  | "certainty_inflation"
  | "overgeneralisation"
  | "framing_effect"
  | "causal_inference_error";

export interface Annotation {
  bias_type: BiasType;
  span_start: number;
  span_end: number;
  flagged_text: string;
  confidence: number;
  severity: Severity;
  clean_alternative: string | null;
  false_positive_check: string | null;
  rag_check_needed: boolean;
  rag_query: string | null;
  conflict: boolean;
  agent_name: string;
  prompt_version: string;
  extras: Record<string, unknown>;
}

export interface AgentRunInfo {
  agent: string;
  bias_type: BiasType;
  prompt_version: string;
  raw_count: number;
  kept_count: number;
  error: string | null;
}

export interface AnalyzeResponse {
  document_id: string;
  mode: Mode;
  overall_bias_score: number;
  annotations: Annotation[];
  agents: AgentRunInfo[];
  warnings: string[];
}

export const BIAS_LABELS: Record<BiasType, string> = {
  confirmation_bias: "Confirmation",
  certainty_inflation: "Certainty inflation",
  overgeneralisation: "Overgeneralisation",
  framing_effect: "Framing",
  causal_inference_error: "Causal inference",
};

export const BIAS_COLORS: Record<BiasType, { bg: string; fg: string; border: string }> = {
  confirmation_bias:    { bg: "#e8f4fe", fg: "#1a6fb5", border: "#93c5fd" },
  certainty_inflation:  { bg: "#fef3e2", fg: "#a0621a", border: "#fbbf24" },
  overgeneralisation:   { bg: "#d1fae5", fg: "#065f46", border: "#6ee7b7" },
  framing_effect:       { bg: "#fce7f3", fg: "#9d174d", border: "#f9a8d4" },
  causal_inference_error: { bg: "#ede9fe", fg: "#4c1d95", border: "#a78bfa" },
};

export const AGENT_BY_BIAS: Record<BiasType, string> = {
  confirmation_bias: "ARGUS",
  certainty_inflation: "LIBRA",
  overgeneralisation: "LENS",
  framing_effect: "QUILL",
  causal_inference_error: "VIGIL",
};
