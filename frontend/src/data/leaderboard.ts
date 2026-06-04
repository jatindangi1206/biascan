// Public model leaderboard data — updated monthly.
// Methodology kept deliberately compact for the website; full details in the paper.
//
// Each row reports two metrics:
//   - catches    : mean ± SD on a bias-injected control text (5 runs)
//                  higher = better at detecting deliberately planted bias
//   - cleanText  : qualitative read of behavior on three real systematic
//                  review variants (5 runs each, methods hidden until paper)

export type Tier =
  | "State of the Art"
  | "Decent"
  | "Budget Proprietary"
  | "Open Source";

export type CleanBehavior =
  | "strict"          // mean ≤ 1.0 across all real texts — applies FP-1 correctly
  | "mostly-clean"    // mean ≤ 3.0
  | "over-flags"      // mean > 3.0 on at least one real text
  | "wrong-order"     // scored a cleaner text higher than a less-clean one
  | "incompatible";   // could not produce parseable JSON

export type Recommendation =
  | "premium"
  | "recommended"
  | "workable"
  | "caution"
  | "not-supported";

export interface LeaderboardRow {
  rank: number;
  model: string;        // openrouter id
  display: string;      // short display name
  vendor: string;
  tier: Tier;
  catches: { mean: number; sd: number };
  cleanText: CleanBehavior;
  recommendation: Recommendation;
}

export const LAST_UPDATED = "2026-05-29";

export const LEADERBOARD: LeaderboardRow[] = [
  { rank: 1, model: "anthropic/claude-opus-4.6",
    display: "Claude Opus 4.6", vendor: "Anthropic",
    tier: "State of the Art",
    catches: { mean: 9.59, sd: 0.58 },
    cleanText: "strict", recommendation: "premium" },

  { rank: 2, model: "openai/gpt-5.4",
    display: "GPT-5.4", vendor: "OpenAI",
    tier: "State of the Art",
    catches: { mean: 9.47, sd: 0.74 },
    cleanText: "strict", recommendation: "premium" },

  { rank: 3, model: "openai/gpt-4.1-nano",
    display: "GPT-4.1 Nano", vendor: "OpenAI",
    tier: "Budget Proprietary",
    catches: { mean: 8.91, sd: 0.05 },
    cleanText: "over-flags", recommendation: "caution" },

  { rank: 4, model: "anthropic/claude-sonnet-4.6",
    display: "Claude Sonnet 4.6", vendor: "Anthropic",
    tier: "State of the Art",
    catches: { mean: 8.73, sd: 0.11 },
    cleanText: "mostly-clean", recommendation: "workable" },

  { rank: 5, model: "deepseek/deepseek-v4-flash",
    display: "DeepSeek V4 Flash", vendor: "DeepSeek",
    tier: "Open Source",
    catches: { mean: 8.69, sd: 0.37 },
    cleanText: "mostly-clean", recommendation: "workable" },

  { rank: 6, model: "google/gemini-2.5-flash-lite",
    display: "Gemini 2.5 Flash-Lite", vendor: "Google",
    tier: "Budget Proprietary",
    catches: { mean: 8.65, sd: 1.90 },
    cleanText: "mostly-clean", recommendation: "workable" },

  { rank: 7, model: "google/gemini-2.5-flash",
    display: "Gemini 2.5 Flash", vendor: "Google",
    tier: "Decent",
    catches: { mean: 8.65, sd: 0.42 },
    cleanText: "over-flags", recommendation: "caution" },

  { rank: 8, model: "anthropic/claude-haiku-4.5",
    display: "Claude Haiku 4.5", vendor: "Anthropic",
    tier: "Decent",
    catches: { mean: 8.58, sd: 1.32 },
    cleanText: "wrong-order", recommendation: "caution" },

  { rank: 9, model: "openai/gpt-4o-mini",
    display: "GPT-4o Mini", vendor: "OpenAI",
    tier: "Budget Proprietary",
    catches: { mean: 7.89, sd: 0.65 },
    cleanText: "over-flags", recommendation: "caution" },

  { rank: 10, model: "qwen/qwen3-235b-a22b",
    display: "Qwen3 235B", vendor: "Alibaba",
    tier: "Open Source",
    catches: { mean: 7.61, sd: 2.08 },
    cleanText: "over-flags", recommendation: "caution" },

  { rank: 11, model: "openai/gpt-4.1-mini",
    display: "GPT-4.1 Mini", vendor: "OpenAI",
    tier: "Budget Proprietary",
    catches: { mean: 7.30, sd: 0.35 },
    cleanText: "strict", recommendation: "recommended" },

  { rank: 12, model: "openai/gpt-4o",
    display: "GPT-4o", vendor: "OpenAI",
    tier: "Decent",
    catches: { mean: 6.97, sd: 0.06 },
    cleanText: "over-flags", recommendation: "caution" },

  { rank: 13, model: "google/gemini-2.5-pro",
    display: "Gemini 2.5 Pro", vendor: "Google",
    tier: "Decent",
    catches: { mean: 6.47, sd: 0.93 },
    cleanText: "mostly-clean", recommendation: "workable" },

  { rank: 14, model: "meta-llama/llama-4-maverick",
    display: "Llama 4 Maverick", vendor: "Meta",
    tier: "Open Source",
    catches: { mean: 5.08, sd: 0.11 },
    cleanText: "over-flags", recommendation: "caution" },

  { rank: 15, model: "qwen/qwen-2.5-72b-instruct",
    display: "Qwen 2.5 72B", vendor: "Alibaba",
    tier: "Open Source",
    catches: { mean: 4.88, sd: 0.00 },
    cleanText: "over-flags", recommendation: "caution" },

  { rank: 16, model: "mistralai/mistral-small-3.1-24b-instruct",
    display: "Mistral Small 3.1 24B", vendor: "Mistral",
    tier: "Open Source",
    catches: { mean: 0.0, sd: 0.0 },
    cleanText: "incompatible", recommendation: "not-supported" },
];
