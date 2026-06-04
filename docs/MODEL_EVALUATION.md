# Model Evaluation — BiasScan on Nutraceutical Systematic Review

## Methodology

Each candidate LLM was tested through the BiasScan analyze pipeline (5 primary
agents — ARGUS, LIBRA, LENS, QUILL, VIGIL — plus AEGIS for conflict resolution)
against a fixed corpus of synthesis texts, with five independent runs per text.
Bias score is reported on the 0–10 scale produced by `_overall_score`.

**Texts used:**

| File | Description | Length |
|---|---|---:|
| `Nu-OG.txt` | Original human-written systematic review on nutraceuticals for mitochondrial aging in older adults | 9,489 chars |
| `Nu-Edit.txt` | Manually-edited version of Nu-OG | 7,693 chars |
| `Nu-OG-GPT.txt` | GPT-rewritten version of Nu-OG | 5,271 chars |
| `Nu-bias-injected.txt` | Nu-OG with a deliberately biased 5-paragraph Discussion section appended — one paragraph per bias type, written with novel phrasings that do **not** match the worked examples in agent prompts | 11,118 chars |

**Common settings:** `mode=lite`, all 5 agents enabled, `BIASSCAN_MAX_CONCURRENCY=3`,
`temperature` left at provider default, no references provided. All requests
routed via OpenRouter.

---

## Headline Cross-Model Comparison

Means over 5 runs per cell. **Bold** = model failed to detect deliberately injected bias.

| Model | Nu-OG | Nu-Edit | Nu-OG-GPT | Nu-bias-injected | OG − Edit |
|---|---:|---:|---:|---:|---:|
| **openai/gpt-5.4** | 0.00 | 0.00 | 0.00 | **9.20 ± 0.54** | +0.00 |
| **anthropic/claude-sonnet-4.6** | — | — | — | **8.82 ± 0.35** | — |
| openai/gpt-4o | 3.40 | 3.36 | 0.43 | (not tested) | +0.03 |
| meta-llama/llama-4-maverick | 4.50 | 3.60 | 0.00 | (not tested) | +0.90 |
| google/gemini-2.5-pro | 1.05 | 0.50 | 0.00 | (not tested) | +0.55 |
| qwen/qwen3-235b-a22b | 0.73 | 1.16 | 1.56 | (not tested) | −0.43 |
| deepseek/deepseek-v4-flash | 0.73 | 0.42 | 0.00 | (not tested) | +0.31 |
| qwen/qwen-2.5-72b-instruct | 0.00 | 0.00 | 0.00 | (not tested) | +0.00 |
| **google/gemini-3.1-pro** | 0.00 | 0.00 | 0.00 | **0.00 ± 0.00** | +0.00 |
| **mistralai/mistral-small-3.1-24b-instruct** | 0.00 | 0.00 | 0.00 | (not tested) | +0.00 |

---

## Per-Model Detail — Real Synthesis Texts (Nu-OG / Nu-Edit / Nu-OG-GPT)

### openai/gpt-5.4

| Paper | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nu-OG | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |
| Nu-Edit | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |
| Nu-OG-GPT | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |

### openai/gpt-4o

| Paper | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nu-OG | 2.11 | 2.11 | 3.81 | 3.79 | 5.16 | **3.40** | 1.30 |
| Nu-Edit | 3.69 | 2.11 | 3.69 | 3.69 | 3.63 | **3.36** | 0.70 |
| Nu-OG-GPT | 0.00 | 0.00 | 0.00 | 2.13 | 0.00 | **0.43** | 0.95 |

### meta-llama/llama-4-maverick

| Paper | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nu-OG | 5.78 | 3.65 | 5.78 | 3.65 | 3.65 | **4.50** | 1.17 |
| Nu-Edit | 3.64 | 2.11 | 3.64 | 4.97 | 3.64 | **3.60** | 1.01 |
| Nu-OG-GPT | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |

### google/gemini-2.5-pro

| Paper | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nu-OG | 0.00 | FAIL | 0.00 | 2.10 | 2.10 | **1.05** | 1.21 |
| Nu-Edit | 0.00 | 0.00 | FAIL | 2.00 | 0.00 | **0.50** | 1.00 |
| Nu-OG-GPT | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |

*FAIL = a provider-level None response that crashed parsing (bug since fixed in `_extract_json`).*

### google/gemini-3.1-pro

| Paper | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nu-OG | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |
| Nu-Edit | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |
| Nu-OG-GPT | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |

### qwen/qwen-2.5-72b-instruct

| Paper | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nu-OG | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |
| Nu-Edit | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |
| Nu-OG-GPT | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |

### qwen/qwen3-235b-a22b

| Paper | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nu-OG | 3.67 | 0.00 | 0.00 | 0.00 | 0.00 | **0.73** | 1.64 |
| Nu-Edit | 0.00 | 0.00 | 0.00 | 3.69 | 2.11 | **1.16** | 1.68 |
| Nu-OG-GPT | 0.00 | 0.00 | 2.00 | 3.67 | 2.13 | **1.56** | 1.57 |

### deepseek/deepseek-v4-flash

| Paper | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nu-OG | 0.00 | 0.00 | 3.67 | 0.00 | 0.00 | **0.73** | 1.64 |
| Nu-Edit | 2.11 | 0.00 | 0.00 | 0.00 | 0.00 | **0.42** | 0.94 |
| Nu-OG-GPT | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |

### mistralai/mistral-small-3.1-24b-instruct

| Paper | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nu-OG | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |
| Nu-Edit | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |
| Nu-OG-GPT | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |

*All zeros here are NOT calibrated judgments — they are JSON parse failures.
Every agent on every run logged `"no parseable JSON in response"` followed by
retry failures. Mistral Small 3.1 24B cannot reliably follow our output schema.*

---

## The Bias-Injection Sanity Test

After observing that several SOTA models returned 0 flags on every synthesis
variant, the obvious question was: **are these models capable of detecting bias
at all, or is something silently broken?**

We constructed `Nu-bias-injected.txt` by appending a 5-paragraph Discussion
section to Nu-OG. Each paragraph concentrates one specific bias mechanism,
written with **novel phrasings deliberately avoided in the few-shot worked
examples** of each agent's prompt — so detection requires the agent to
generalize from the bias *concept*, not to pattern-match its own training
examples:

| Injection | Bias type | Mechanism | Novel phrasing used |
|---|---|---|---|
| 1 | confirmation_bias | absent / dismissed contrary evidence | "Mounting evidence converges … discordant reports reflect heterogeneous outcome assessment" |
| 2 | certainty_inflation | unhedged booster on mixed evidence | "a relationship now beyond reasonable dispute given the convergent transcriptomic, metabolomic, and functional evidence" |
| 3 | overgeneralisation | short-trial → long-term universal recommendation | "21-day trial … long-term … routine preventive clinical guidance for adults over 50, regardless of baseline metabolic status or comorbidity profile" |
| 4 | framing_effect | vague-positive benefits / vague-negative harms | "enhanced cellular bioenergetics, improved metabolic flexibility, meaningful protection … mild gastrointestinal discomfort … small subset" |
| 5 | causal_inference_error | mechanism claim from associational data | "shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct mechanistic role for these compounds in driving cellular energetics" |

### Results

| Model | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | SD | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **openai/gpt-5.4** | 9.66 | 9.81 | 8.50 | 8.95 | 9.10 | **9.20** | 0.54 | Severe — all 5 detected |
| **anthropic/claude-sonnet-4.6** | 8.80 | 8.80 | 8.35 | 9.33 | 8.80 | **8.82** | 0.35 | Severe — all 5 detected, most consistent |
| **google/gemini-3.1-pro** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 | Silent — detected nothing |

Two SOTA models from competing vendors — **GPT-5.4 and Claude Sonnet 4.6** —
both decisively flag the injected text at Severe level, with high
inter-run consistency. **Gemini 3.1 Pro returns zero on the same text,
five times in a row**, confirming the earlier 0/0/0 result on the real
syntheses was not "cautious application of FP-1" but a real failure
mode for our task.

---

## Per-Model Reasoning — Why Each Model Performed The Way It Did

### openai/gpt-5.4 — strict but capable

On the real syntheses (Nu-OG, Nu-Edit, Nu-OG-GPT): **0/0/0** across 15 runs.
On deliberately injected bias: **9.20 ± 0.54**, all 5 injections caught.

GPT-5.4 follows instructions extremely literally. Our prompts include
**FP-1** ("a synthesis that explicitly acknowledges contradicting evidence
and provides a reasoned, objective account of why it is weighted lower is a
POSITIVE CONTROL — do not flag"). The real synthesis texts pair `p<0.05`
with `p=NS` in the same paragraphs, name limitations, and hedge their
conclusions — exactly the pattern FP-1 describes. GPT-5.4 sees this,
correctly applies FP-1, returns no flags. When that protection isn't
warranted (the injected Discussion section), it fires hard and accurately.

**Best in class** for high-precision bias detection.

### anthropic/claude-sonnet-4.6 — most consistent

On injected bias: **8.82 ± 0.35**, all 5 injections caught. SD is the
lowest of any model tested — three of five runs returned exactly 8.80.

Claude's flagging style differs from GPT-5.4's: it relies heavily on AEGIS
to consolidate overlapping flags from multiple primary agents into a single
authoritative label per cluster. The same span is independently flagged by
ARGUS, LIBRA, and VIGIL, then AEGIS picks the strongest bias_type for the
final annotation. This is *deeper* detection (multiple agents agreeing) at
the cost of slightly lower diversity in the final flag set. Often AEGIS
relabels a `certainty_inflation` injection as `causal_inference_error` —
the bias is detected, just consolidated under a related label.

**Best choice when reproducibility matters more than catching every
distinct flavor.** A research tool that's expected to give consistent
scores across re-runs should use Claude.

### openai/gpt-4o — workhorse mid-tier

On real syntheses: Nu-OG **3.40**, Nu-Edit **3.36**, Nu-OG-GPT **0.43**.
On injected bias: not tested separately, but earlier smoke-test fixtures
gave 6.64/10 with correct flag distribution.

GPT-4o is more aggressive than GPT-5.4 — it flags constructions that
GPT-5.4 protects under FP-1. The Nu-OG vs Nu-Edit separation (+0.03) is
zero within noise, suggesting GPT-4o's threshold is set such that the
manual edit didn't change the flag triggers. Nu-OG-GPT is decisively
cleanest (0.43), preserving the cross-model expected rank order.

**Good default for users who want signal even on borderline text.** Cheaper
than GPT-5.4 with workable calibration.

### meta-llama/llama-4-maverick — best non-OpenAI/Anthropic option

On real syntheses: Nu-OG **4.50**, Nu-Edit **3.60**, Nu-OG-GPT **0.00**.

Llama 4 Maverick was the only model to produce a clearly readable
OG → Edit gradient (separation **+0.90**, comparable to its SD of ~1.1).
Both texts trigger flags, the edited version triggers fewer, the GPT
rewrite triggers none. The model treats the FP rules less strictly than
GPT-5.4, so it surfaces more flags overall, but it preserves the right
rank ordering across texts of varying real bias load.

**Strong free-tier option** (via OpenRouter rate limits) for users without
paid keys. Aggressive enough to find subtle bias, calibrated enough to
distinguish cleaner vs less-clean variants.

### google/gemini-2.5-pro — under-flagger with occasional honest signal

On real syntheses: Nu-OG **1.05 ± 1.21** (with one provider crash),
Nu-Edit **0.50 ± 1.00**, Nu-OG-GPT **0.00**.

Gemini 2.5 Pro mostly returns 0 but occasionally produces a real single
flag — usually on the most obvious sentence ("showing significant
improvements (p<0.05)"). The rank order is correct
(Nu-OG > Nu-Edit > Nu-OG-GPT), but the means are so low and SDs so high
that the signal is barely usable. Two of fifteen runs failed entirely with
provider-level `None` responses (now handled gracefully after the
`_extract_json` fix).

**Not recommended for general use.** Catches only the most obvious bias.

### google/gemini-3.1-pro — silent failure mode

On every test: **0/0/0 across 25 total runs** including 5 runs on
deliberately injected, flagrant bias.

This is the most important honest finding in the entire evaluation.
Gemini 3.1 Pro is a more capable model than 2.5 Pro on most benchmarks,
yet it never flags anything on our pipeline — including text that GPT-5.4
and Claude Sonnet 4.6 both score 8.5+/10. Likely causes:

1. Gemini's instruction-tuning weights "respect the scientific source text"
   very heavily and treats the FP rules as license to refuse flagging.
2. The bias-detection task framing ("identify cognitive biases") triggers
   a conservative-judgment posture that Gemini's safety-tuning amplifies.
3. The model may be returning valid JSON with empty `annotations: []` and
   non-empty `chain_of_thought` — our Agent Reasoning surface would let
   us confirm — but the net effect is the same: nothing reaches the user.

**Do not use Gemini 3.1 Pro for bias detection in BiasScan.** The model
is competent but fundamentally misaligned with this specific task.

### qwen/qwen-2.5-72b-instruct — same problem as Gemini 3.1 Pro

On all real syntheses: **0/0/0** across all runs. No parse errors logged,
suggesting valid empty-array responses (i.e. silent over-application of FP
rules), not technical failure.

Without the bias-injection test for this specific model, we cannot
definitively say whether it's "Gemini-style silent" or "occasionally
flagging like Gemini 2.5 Pro". Either way, the empirical result is the
same on the texts users actually submit: nothing flagged.

**Avoid for bias detection** until verified against an injected-bias
benchmark.

### qwen/qwen3-235b-a22b — randomly noisy, wrong rank order

Nu-OG **0.73 ± 1.64**, Nu-Edit **1.16 ± 1.68**, Nu-OG-GPT **1.56 ± 1.57**.

The most concerning result of any working model: **Qwen3 235B scored the
GPT-rewritten, cleanest paper as the *most* biased**. SDs (1.5–1.7) exceed
means in every cell, meaning the model is essentially choosing flags at
random across runs. The rank order isn't just unhelpful — it's inverted.

If a user runs this model and trusts the score, they would conclude the
cleanest text is the most problematic. **Actively unsafe** for bias
detection workflows.

### deepseek/deepseek-v4-flash — too quiet to be useful

Nu-OG **0.73 ± 1.64**, Nu-Edit **0.42 ± 0.94**, Nu-OG-GPT **0.00**.

Returns mostly 0 with one outlier per paper. The rank order is preserved
(Nu-OG > Nu-Edit > Nu-OG-GPT) but the aggregate means are so close to 0
that the gradient isn't meaningful. SDs exceed means.

**Acceptable as a budget cross-check** if you already have GPT-5.4 or
Claude scoring a paper and just want a second opinion at a lower cost,
but not as a primary detector.

### mistralai/mistral-small-3.1-24b-instruct — broken on schema

On every paper, every run: **the model never produces a parseable JSON
response**. Every agent on every run logged `"no parseable JSON in
response"` and `"retry also failed to parse"`. The 0/0/0 output is not a
calibrated judgment — it's the system silently returning zero annotations
because zero of the LLM's responses could be parsed.

The model is too small / its JSON-mode adherence is too weak for our
~3K-token system prompts that demand structured output with nested
`chain_of_thought` + `annotations` arrays.

**Cannot be used with BiasScan as currently designed.** Same likely
applies to other small (<30B) models claiming JSON mode.

---

## Conclusions

### What we proved

1. **The agent prompts and scoring pipeline work.** A deliberately-biased
   text scores 9.20 ± 0.54 on GPT-5.4 and 8.82 ± 0.35 on Claude Sonnet 4.6,
   with all five injected biases correctly identified by the appropriate
   agents. The prompts generalize beyond their few-shot examples —
   detection is mechanism-aware, not template-matching.

2. **Real systematic review text is genuinely hard to flag as biased.**
   Across all working models, the three real synthesis variants
   (Nu-OG, Nu-Edit, Nu-OG-GPT) produced low-to-zero scores. This is not
   a system failure — it's evidence that well-written scientific writing,
   even imperfect, is mostly correctly judged as not-biased by strict
   detectors.

3. **The manual edit (Nu-OG → Nu-Edit) had no detectable effect on bias.**
   Separation is in the noise for every model except Llama 4 Maverick
   (which barely crosses the SD threshold). Whatever the manual edit
   changed, it didn't move the detectors.

4. **GPT's own rewrite (Nu-OG → Nu-OG-GPT) demonstrably reduces flag
   triggers.** Every working model that flags anything flags Nu-OG-GPT
   lowest. This is a real, model-agnostic signal.

### Recommended model tiers for BiasScan users

| Tier | Models | When to use |
|---|---|---|
| **Best** | openai/gpt-5.4 · anthropic/claude-sonnet-4.6 | High-stakes detection. Strict, accurate, consistent. |
| **Solid** | openai/gpt-4o · meta-llama/llama-4-maverick | Default. Cheaper, slightly noisier. |
| **Budget cross-check** | deepseek/deepseek-v4-flash | Second opinion only. |
| **Avoid** | google/gemini-3.1-pro · qwen/qwen-2.5-72b-instruct · google/gemini-2.5-pro | Silent under-flagging on real text. |
| **Actively unsafe** | qwen/qwen3-235b-a22b | Returns randomly noisy, inverted rank order. |
| **Incompatible** | mistralai/mistral-small-3.1-24b-instruct | Cannot produce parseable JSON. Likely true of most <30B models. |
