# BiasScan Evaluation Protocol

Current protocol used for the latest merged benchmark in
`eval/output/leaderboard.json`.

This note is written to answer the specific questions that came up during
review:

1. How are we injecting bias?
2. What is the protocol for runs?
3. What is the comparative ground truth?
4. Why are there API failures?
5. What are the scoring parameters, and if LLMs are involved, how are they evaluated?

## 1. What exactly is being evaluated?

We are not evaluating general chatbot ability.

We hold the BiasScan pipeline fixed and vary the underlying LLM.

So the object of evaluation is:

how the full BiasScan system behaves when powered by different models.

The fixed system components are:

- ARGUS: confirmation bias
- LIBRA: certainty inflation
- LENS: overgeneralisation
- QUILL: framing effect
- VIGIL: causal inference error
- AEGIS: overlap/conflict resolver

The practical question is:

when only the LLM changes, does BiasScan:

1. stay low on cleaner synthesis text,
2. rise on deliberately biased text,
3. behave consistently across reruns, and
4. avoid technical failure?

So we are benchmarking model choice inside BiasScan, not claiming a general
measure of model intelligence.

## 2. How bias is injected

We use `backend/app/synthesis/Nu-OG.txt` as the base document.

From that base, we created:

- `Nu-Edit.txt`: a manually cleaned version
- `Nu-OG-GPT.txt`: a GPT-rewritten version
- `Nu-bias-injected.txt`: the same base text plus a deliberately biased
  Discussion section

The positive-control document is constructed by appending 5 new paragraphs,
each designed to express one target bias:

1. confirmation bias  
   Contradictory/null evidence is mentioned and then discounted as measurement
   noise or methodological irrelevance.

2. certainty inflation  
   Mixed evidence is described as settled or beyond reasonable dispute.

3. overgeneralisation  
   Narrow or short-duration findings are extended into broad clinical guidance.

4. framing effect  
   Benefits are described positively and prominently, while harms are softened
   or minimised.

5. causal inference error  
   Biomarker or associative findings are described as direct mechanistic proof.

Important design choice:
The injected paragraphs use novel wording rather than copying the worked
examples from the agent prompts. The goal is to test conceptual
generalisation, not prompt-example matching.

### Prompt used to generate the injected text

The exact original generation prompt was not stored separately in the
repository. The following is a faithful reconstruction based on the final
stored text in `Nu-bias-injected.txt`:

```text
Take the following scientific synthesis and append a new section titled
"Discussion and Clinical Implications".

Write exactly 5 new paragraphs. Each paragraph should introduce one specific
bias mechanism while still sounding like plausible academic systematic-review
prose:

1. confirmation bias: emphasize supportive evidence and dismiss
discordant/null findings as heterogeneity, technical noise, or lower-value
evidence
2. certainty inflation: turn mixed or limited evidence into a highly certain
conclusion
3. overgeneralisation: extrapolate short-term or narrow findings into a broad
long-term clinical recommendation for adults over 50 regardless of baseline
status or comorbidity profile
4. framing effect: describe benefits in strong positive terms while
minimizing harms briefly and vaguely
5. causal inference error: convert associational biomarker findings into a
direct mechanistic causal claim

Requirements:
- append only; do not rewrite the original synthesis
- keep the tone formal and journal-like
- use novel wording, not obvious caricature
- make the bias detectable but still realistic
- do not label the paragraphs by bias type
```

## 3. Protocol for runs

### Materials

Each model is run on the same 4 texts:

1. `Nu-OG`
2. `Nu-OG-GPT`
3. `Nu-Edit`
4. `Nu-bias-injected`

### Common settings

All models are run with the same settings:

- same BiasScan codebase
- same prompt version
- all 5 primary agents enabled
- AEGIS enabled
- `mode=lite`
- no references supplied
- same routing path (`openrouter`)

### Repetition

For the latest merged benchmark, the stored artifact uses:

- 18 models
- 4 texts per model
- 2 runs per model per text

Important note:
Some earlier notes mention 5-run pilot experiments. Those were exploratory.
The latest merged canonical artifact in `eval/output/leaderboard.json`
currently stores `n_runs = 2`, and that is the number that should be cited for
the present benchmark.

### Step-by-step run procedure

For each candidate model:

1. Build provider config for that model.
2. Load one synthesis text.
3. Run the full BiasScan pipeline.
4. Repeat the run `n_runs` times under the same settings.
5. Store:
   - per-run score
   - number of flags
   - returned flags
   - elapsed time
   - warnings/errors when available
6. Aggregate mean and SD for that model-text cell.
7. Repeat for all 4 texts.
8. Repeat for all models.

### Reruns

Some rows that looked suspiciously like silent zeros were rerun separately using
`eval/rerun_excluded.py`.

That rerun protocol adds warning capture and `api_failure` tagging, then merges
the corrected rows back into the main leaderboard.

## 4. Comparative ground truth

Current ground truth is mixed. It is not yet a full human-adjudicated gold
standard.

### Strongest ground truth

The strongest current ground truth is `Nu-bias-injected.txt`, because we know
exactly which bias mechanisms we inserted and where they were inserted at the
paragraph level.

### Comparative controls

The other three texts serve as relative controls:

- `Nu-Edit` should be cleaner than `Nu-OG`
- `Nu-OG-GPT` is a rewrite control
- `Nu-OG` is the natural baseline

These are useful for comparative ranking, but they are not expert span-level
gold labels.

### Current limitation

We do not yet have expert human annotation for every biased span in the natural
texts. So at present this is a controlled benchmarking setup, not a final
validation study with adjudicated real-world ground truth.

## 5. Why there are API failures

Not all zero scores are real zeros.

We observed several kinds of technical failure:

1. Invalid model IDs  
   Example: some Gemini model IDs were rejected by OpenRouter as not valid.

2. Provider-side failures  
   Timeouts, rate limits, or other upstream API errors.

3. Schema failures  
   The model returns text that cannot be parsed as the required JSON structure.

4. Span anchoring failures  
   A model returns a flag, but the returned span cannot be reliably mapped back
   to the original text.

5. Parse failures after retry  
   The pipeline retries once on total parse failure; some models still fail.

### How we handle this

In the rerun protocol, if 3 or more agents fail with provider-like errors in a
single run, that run is marked as `api_failure`.

That means we do not treat that row as a genuine “the model thinks this text is
clean” result.

This matters especially for:

- `google/gemini-3-flash`
- `google/gemini-3.1-pro`

Those are invalid-ID failures in the final rerun-enhanced artifact, not true
negative detections.

## 6. What prompt was used

This is not one single prompt. It is a fixed prompt stack.

### System prompts

We use the same versioned system prompts for every model:

- `backend/app/prompts/v1/argus_v1.0.txt`
- `backend/app/prompts/v1/libra_v1.0.txt`
- `backend/app/prompts/v1/lens_v1.0.txt`
- `backend/app/prompts/v1/quill_v1.0.txt`
- `backend/app/prompts/v1/vigil_v1.0.txt`
- `backend/app/prompts/v1/aegis_v1.0.txt`

Each one defines:

- the bias type
- the false-positive rules
- the reasoning protocol
- the expected JSON schema

### User message template

Each primary agent receives the same user-message format:

```text
=== MODE === lite

=== SYNTHESIS TEXT (analyse this) ===
[full synthesis text]

=== REFERENCE LIST (for context) ===
[No reference list provided.]

Apply your reasoning protocol. Return ONLY the JSON object with key
'annotations'. Character offsets must index into the SYNTHESIS TEXT exactly as given.
```

### Parse retry rule

If the model fails to return parseable JSON, the pipeline retries once with an
explicit correction telling it to return only one valid JSON object and no
extra prose.

## 7. What the LLM does vs what the scoring code does

This is important.

The candidate LLM is used to generate annotations.
The final document score is not assigned by another judge LLM.

The score is computed deterministically in code after the annotations are
returned.

So there are two separate layers:

1. LLM layer  
   Produces candidate annotations.

2. Scoring layer  
   Converts surviving annotations into a document-level score.

## 8. Scoring parameters

### Stage A: annotation generation

Each agent outputs structured annotations with:

- `bias_type`
- `flagged_text`
- `span_start`
- `span_end`
- `certainty` or `confidence`
- `severity`

### Stage B: certainty mapping

If the model uses the discrete `certainty` field, we map it as follows:

- `certain` = 1.0
- `probable` = 0.8
- `suspected` = 0.6
- `weak` = 0.4

### Stage C: confidence floor

Annotations with confidence below `0.5` are dropped.

### Stage D: evidence cross-check

Evidence RAG can adjust confidence up or down by up to `±0.15`.

This is a small refinement step. It does not replace the model judgment.

### Stage E: AEGIS conflict resolution

If overlapping spans are flagged as different bias types, AEGIS resolves the
conflict and keeps the best consolidated annotation.

### Stage F: final score

Let:

- `n` = number of surviving annotations
- `high_conf_sum` = sum of confidence for high-severity flags
- `medium_conf_sum` = sum of confidence for medium-severity flags
- `unique_types` = number of distinct bias types

The current score formula is:

- `base = min(8.0, 12*n / (5+n))`
- `severity_bump = min(1.5, 0.4*high_conf_sum + 0.15*medium_conf_sum)`
- `diversity_bump = 0.15 * (unique_types - 1)`
- `internal_score = base + severity_bump + diversity_bump`
- final score = `min(10, internal_score) / 10`

The benchmark table reports that final score on a 0-10 scale.

### Interpretation

This means the score is mainly driven by:

1. number of detected flags
2. severity/confidence of those flags
3. diversity of bias types

It is not a second LLM saying “this paper is 8/10 biased.”

## 9. How models are ranked

In the current benchmark, ranking is driven primarily by performance on the
positive-control injected text.

Operationally, the desirable pattern is:

- high score on `Nu-bias-injected`
- low score on `Nu-Edit`, `Nu-OG-GPT`, and ideally `Nu-OG`
- low variance across runs
- no invalid-ID or technical failure issues

So the ranking is trying to reward:

1. sensitivity to real bias signals
2. specificity on cleaner controls
3. stability across reruns
4. technical reliability

## 10. What this protocol can and cannot currently claim

### What it can claim

It can compare models within the BiasScan pipeline under controlled conditions.

It can tell us which models:

- detect the injected biases strongly,
- over-flag cleaner texts,
- behave inconsistently,
- or fail technically.

### What it cannot yet claim

It does not yet establish gold-standard real-world accuracy on natural
scientific synthesis, because we do not yet have expert-adjudicated span-level
ground truth for the natural texts.

## 11. Current limitations

The main limitations of the current protocol are:

1. small corpus
2. single domain focus
3. synthetic positive-control bias
4. no full human-adjudicated span labels on natural texts
5. latest canonical merged benchmark stores only 2 runs per cell

## 12. Planned next step for a stronger research protocol

To make this publication-grade, the next upgrade should be:

1. expert human annotation of real synthesis texts
2. inter-rater agreement measurement
3. larger multi-domain corpus
4. more repeated runs per cell
5. pre-specified criteria for sensitivity, specificity, and stability

## 13. Short answer

The current evaluation is a controlled system benchmark:

- we inject known bias into one positive-control document,
- run every model through the same multi-agent BiasScan pipeline,
- compare scores on clean vs biased texts,
- separate technical failures from genuine zero detections,
- and rank models by sensitivity, specificity, consistency, and reliability.
