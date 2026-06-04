# BiasScan Detailed Leaderboard Report

## Canonical storage

- Latest merged benchmark artifact: `eval/output/leaderboard.json` (modified 2026-05-29 18:02:41)
- Rerun-only artifact: `eval/output/leaderboard_rerun.json` (modified 2026-05-29 18:02:41)
- This benchmark's per-run detail is stored inline under `results[*].papers[*].runs[*]` in `leaderboard.json`.
- There is no deeper persisted per-agent trace archive for this specific benchmark in `eval/results/`; that directory is empty for the latest run.
- `docs/MODEL_EVALUATION.md` is a narrative write-up, not the canonical latest raw data file.

## Coverage

- Papers: `Nu-OG, Nu-OG-GPT, Nu-Edit, Nu-bias-injected`
- Runs per paper in the stored artifact: `2`
- Models in the latest merged leaderboard: `18`
- Warning-level metadata such as `api_failure`, `n_agent_errors`, and `warnings` exists only for the five rerun models that were merged back in from `leaderboard_rerun.json`.

## Summary leaderboard

| Rank | Model | Tier | Nu-OG | Nu-OG-GPT | Nu-Edit | Nu-bias-injected | Status |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `anthropic/claude-opus-4.6` | State of the Art | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 9.59±0.58 | OK |
| 2 | `openai/gpt-5.4` | State of the Art | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 9.47±0.74 | OK |
| 3 | `openai/gpt-4.1-nano` | Budget Proprietary | 6.34±0.64 | 5.81±5.23 | 6.53±1.12 | 8.91±0.05 | OK |
| 4 | `anthropic/claude-sonnet-4.6` | State of the Art | 1.81±2.57 | 1.82±2.57 | 0.00±0.00 | 8.73±0.11 | OK |
| 5 | `deepseek/deepseek-v4-flash` | Open Source | 1.05±1.48 | 0.00±0.00 | 0.00±0.00 | 8.69±0.37 | OK |
| 6 | `google/gemini-2.5-flash-lite` | Budget Proprietary | 1.83±2.60 | 0.00±0.00 | 0.00±0.00 | 8.65±1.90 | OK |
| 7 | `google/gemini-2.5-flash` | Decent | 4.75±3.73 | 4.63±0.18 | 2.11±0.00 | 8.65±0.42 | OK |
| 8 | `anthropic/claude-haiku-4.5` | Decent | 1.72±2.43 | 4.20±0.93 | 4.59±1.50 | 8.58±1.32 | OK |
| 9 | `openai/gpt-4o-mini` | Budget Proprietary | 7.76±0.54 | 6.33±0.29 | 2.13±0.00 | 7.89±0.65 | OK |
| 10 | `qwen/qwen3-235b-a22b` | Open Source | 0.00±0.00 | 2.84±4.02 | 0.00±0.00 | 7.61±2.08 | OK |
| 11 | `openai/gpt-4.1-mini` | Budget Proprietary | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 7.30±0.35 | OK |
| 12 | `openai/gpt-4o` | Decent | 5.05±0.08 | 0.00±0.00 | 2.90±1.12 | 6.97±0.06 | OK |
| 13 | `google/gemini-2.5-pro` | Decent | 2.10±0.00 | 0.00±0.00 | 0.00±0.00 | 6.47±0.93 | OK |
| 14 | `meta-llama/llama-4-maverick` | Open Source | 5.74±0.05 | 0.00±0.00 | 4.25±0.87 | 5.08±0.11 | OK |
| 15 | `qwen/qwen-2.5-72b-instruct` | Open Source | 7.77±1.68 | 2.90±1.09 | 0.00±0.00 | 4.88±0.00 | OK |
| 16 | `mistralai/mistral-small-3.1-24b-instruct` | Open Source | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | OK |
| · | `google/gemini-3-flash` | Budget Proprietary | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | INVALID_ID |
| · | `google/gemini-3.1-pro` | State of the Art | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | INVALID_ID |

## Per-model full detail

### anthropic/claude-opus-4.6

- Tier: `State of the Art`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 0.00 | 0.00 | 2 | 26.30 |
| Nu-OG-GPT | 0.00 | 0.00 | 2 | 26.10 |
| Nu-Edit | 0.00 | 0.00 | 2 | 28.40 |
| Nu-bias-injected | 9.59 | 0.58 | 2 | 88.20 |

#### Nu-OG

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `26.30` seconds
- `api_failures`: `0`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 5 segments.
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 5 segments.
- `flags`: none

#### Nu-OG-GPT

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `26.10` seconds
- `api_failures`: `0`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 3 segments.
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 3 segments.
- `flags`: none

#### Nu-Edit

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `28.40` seconds
- `api_failures`: `0`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 4 segments.
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 4 segments.
- `flags`: none

#### Nu-bias-injected

- Aggregate: mean `9.59`, sd `0.58`, valid `2`, elapsed `88.20` seconds
- `api_failures`: `0`

#### Run 1

- `ok`: `True`
- `score`: `9.18`
- `n_flags`: `8`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 10 segments.
- `flags`:
  - AEGIS · confirmation_bias · high · conf `1.000` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - AEGIS · overgeneralisation · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · causal_inference_error · high · conf `1.000` · "n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now beyond reasonable dispute give"
  - AEGIS · confirmation_bias · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · overgeneralisation · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · confirmation_bias · high · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - AEGIS · causal_inference_error · high · conf `0.800` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - AEGIS · causal_inference_error · high · conf `1.000` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

#### Run 2

- `ok`: `True`
- `score`: `10.00`
- `n_flags`: `12`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 10 segments.
- `flags`:
  - QUILL · framing_effect · medium · conf `0.520` · "Rossman et al. assessed mitochondrial-specific ROS-bioactivity showing a significant decrease (–35%; P<0.05) after sodium nitrite supplement"
  - ARGUS · confirmation_bias · medium · conf `0.881` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - AEGIS · confirmation_bias · high · conf `1.000` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - AEGIS · certainty_inflation · high · conf `0.800` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · overgeneralisation · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · confirmation_bias · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · overgeneralisation · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - AEGIS · overgeneralisation · high · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - AEGIS · causal_inference_error · high · conf `1.000` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"
  - AEGIS · overgeneralisation · medium · conf `0.800` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

### openai/gpt-5.4

- Tier: `State of the Art`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 0.00 | 0.00 | 2 | 3.40 |
| Nu-OG-GPT | 0.00 | 0.00 | 2 | 2.90 |
| Nu-Edit | 0.00 | 0.00 | 2 | 3.20 |
| Nu-bias-injected | 9.47 | 0.74 | 2 | 35.10 |

#### Nu-OG

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `3.40` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-OG-GPT

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `2.90` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-Edit

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `3.20` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-bias-injected

- Aggregate: mean `9.47`, sd `0.74`, valid `2`, elapsed `35.10` seconds

#### Run 1

- `ok`: `True`
- `score`: `10.00`
- `n_flags`: `10`
- `flags`:
  - AEGIS · confirmation_bias · high · conf `1.000` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - LIBRA · certainty_inflation · medium · conf `0.725` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - AEGIS · overgeneralisation · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · confirmation_bias · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · overgeneralisation · high · conf `1.000` · "long-term nicotinamide riboside supplementation should be incorporated into routine preventive clinical guidance for adults over 50 seeking"
  - LENS · overgeneralisation · high · conf `1.000` · "regardless of baseline metabolic status or comorbidity profile"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - QUILL · framing_effect · high · conf `0.850` · "Mild gastrointestinal discomfort was reported in a small subset of participants."
  - AEGIS · causal_inference_error · high · conf `1.000` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"
  - VIGIL · causal_inference_error · medium · conf `0.719` · "driving cellular energetics"

#### Run 2

- `ok`: `True`
- `score`: `8.95`
- `n_flags`: `7`
- `flags`:
  - AEGIS · causal_inference_error · medium · conf `1.000` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - ARGUS · confirmation_bias · high · conf `0.995` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - AEGIS · overgeneralisation · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - QUILL · framing_effect · high · conf `0.650` · "Mild gastrointestinal discomfort was reported in a small subset of participants."
  - AEGIS · causal_inference_error · high · conf `1.000` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

### openai/gpt-4.1-nano

- Tier: `Budget Proprietary`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 6.34 | 0.64 | 2 | 5.40 |
| Nu-OG-GPT | 5.81 | 5.23 | 2 | 11.10 |
| Nu-Edit | 6.53 | 1.12 | 2 | 5.90 |
| Nu-bias-injected | 8.91 | 0.05 | 2 | 22.90 |

#### Nu-OG

- Aggregate: mean `6.34`, sd `0.64`, valid `2`, elapsed `5.40` seconds

#### Run 1

- `ok`: `True`
- `score`: `5.89`
- `n_flags`: `4`
- `flags`:
  - LIBRA · certainty_inflation · medium · conf `0.950` · "significant improvements (p<0.05)"
  - LIBRA · certainty_inflation · medium · conf `0.950` · "significant improvements"
  - LIBRA · certainty_inflation · medium · conf `0.877` · "significant differences"
  - LIBRA · certainty_inflation · medium · conf `0.950` · "significant reduction"

#### Run 2

- `ok`: `True`
- `score`: `6.79`
- `n_flags`: `5`
- `flags`:
  - LIBRA · certainty_inflation · medium · conf `0.950` · "significant improvements (p<0.05)"
  - ARGUS · confirmation_bias · medium · conf `0.724` · "no significant differences between groups were underlined"
  - LIBRA · certainty_inflation · medium · conf `0.714` · "significant decrease (–35%; P<0.05)"
  - LIBRA · certainty_inflation · medium · conf `0.950` · "significant reduction in plasma acylcarnitine levels (p<0.05) and a significant reduction in ceramide levels (p<0.05)"
  - LIBRA · certainty_inflation · medium · conf `0.950` · "significant improvements in gene expression of UCP3 (~30%, p<0.05) and UQCRC1 (~20%, p<0.05)"

#### Nu-OG-GPT

- Aggregate: mean `5.81`, sd `5.23`, valid `2`, elapsed `11.10` seconds

#### Run 1

- `ok`: `True`
- `score`: `9.51`
- `n_flags`: `9`
- `flags`:
  - LIBRA · certainty_inflation · medium · conf `0.950` · "Study Identification and Se"
  - QUILL · framing_effect · high · conf `1.000` · "y files"
  - QUILL · framing_effect · high · conf `0.850` · "32 days"
  - QUILL · framing_effect · high · conf `0.850` · "21 days"
  - QUILL · framing_effect · high · conf `0.850` · "four months"
  - QUILL · framing_effect · high · conf `0.913` · "six months"
  - QUILL · framing_effect · high · conf `0.850` · "12 weeks"
  - VIGIL · causal_inference_error · high · conf `0.518` · "significantly improved"
  - VIGIL · causal_inference_error · high · conf `0.520` · "reduced circulating"

#### Run 2

- `ok`: `True`
- `score`: `2.11`
- `n_flags`: `1`
- `flags`:
  - LIBRA · certainty_inflation · medium · conf `0.750` · "Study Identifica"

#### Nu-Edit

- Aggregate: mean `6.53`, sd `1.12`, valid `2`, elapsed `5.90` seconds

#### Run 1

- `ok`: `True`
- `score`: `7.32`
- `n_flags`: `6`
- `flags`:
  - ARGUS · confirmation_bias · medium · conf `0.724` · "reported in Table 1."
  - LIBRA · certainty_inflation · medium · conf `0.799` · "ch water, nicotinami"
  - LIBRA · certainty_inflation · medium · conf `0.878` · "reported statistically significant within-group increases"
  - LIBRA · certainty_inflation · medium · conf `0.875` · "reported a statistically significant reduction"
  - LIBRA · certainty_inflation · low · conf `1.000` · "did not identify statistically significant differences"
  - LIBRA · certainty_inflation · medium · conf `0.878` · "reported statistically significant increases"

#### Run 2

- `ok`: `True`
- `score`: `5.74`
- `n_flags`: `4`
- `flags`:
  - LIBRA · certainty_inflation · medium · conf `0.675` · "statistically significant"
  - LIBRA · certainty_inflation · medium · conf `0.675` · "significant"
  - LIBRA · certainty_inflation · medium · conf `0.675` · "significant"
  - LIBRA · certainty_inflation · medium · conf `0.675` · "significant"

#### Nu-bias-injected

- Aggregate: mean `8.91`, sd `0.05`, valid `2`, elapsed `22.90` seconds

#### Run 1

- `ok`: `True`
- `score`: `8.87`
- `n_flags`: `12`
- `flags`:
  - QUILL · framing_effect · high · conf `1.000` · "50%"
  - QUILL · framing_effect · low · conf `1.000` · "r a total of 201 he"
  - QUILL · framing_effect · low · conf `0.921` · "ncluded in the systematic review. The"
  - QUILL · framing_effect · low · conf `0.850` · "jects included ranged from 67.0±1."
  - QUILL · framing_effect · low · conf `1.000` · "from 67.0±1.0 years to 76.0±5.6 years"
  - QUILL · framing_effect · low · conf `1.000` · "years. It should be noticed that Yoshino e"
  - QUILL · framing_effect · low · conf `1.000` · "terize the study sample for gender differences."
  - QUILL · framing_effect · low · conf `0.850` · "n the remaining 181 participants, the sample of th"
  - QUILL · framing_effect · low · conf `1.000` · "83 males and 98 females"
  - QUILL · framing_effect · low · conf `0.923` · "BMI ranging from 25.3±1.3 kg/ m2 to 28.6±3.9 kg/m2"
  - QUILL · framing_effect · low · conf `0.930` · "4 studies did not report"
  - AEGIS · causal_inference_error · high · conf `0.800` · "long-term nicotinamide riboside supplementation should be incorporated into routine preventive clinical guidance"

#### Run 2

- `ok`: `True`
- `score`: `8.94`
- `n_flags`: `17`
- `flags`:
  - QUILL · framing_effect · low · conf `1.000` · "Three studies (50%) were conducted in the USA"
  - QUILL · framing_effect · low · conf `0.850` · "sample size of the RCTs included ranged from 12 to 66"
  - QUILL · framing_effect · low · conf `0.850` · "total of 201 healthy older adults included"
  - QUILL · framing_effect · low · conf `1.000` · "ults included in the systematic review. The"
  - QUILL · framing_effect · low · conf `0.920` · "be noticed that Yoshino et al. did not characterize t"
  - QUILL · framing_effect · low · conf `0.850` · "t, based on the remaining 181 participants, the sample of the present study was composed of 83"
  - QUILL · framing_effect · low · conf `0.925` · "sample of the present study was composed of 83 males and 98 females"
  - QUILL · framing_effect · low · conf `1.000` · "males and 98 females, with a body composition assessed by BMI"
  - QUILL · framing_effect · low · conf `0.923` · "BMI ranging from 25.3±1.3 kg/ m2 to 28.6±3.9 kg/m2"
  - QUILL · framing_effect · low · conf `0.923` · "ranging from 25.3±1.3 kg/ m2 to 28.6±3.9 kg/m2. Interestingly, 4 studies"
  - QUILL · framing_effect · low · conf `0.924` · "did not report any standardization in terms of diet and physical"
  - QUILL · framing_effect · low · conf `0.921` · "activity during the study protocol, while both Zanini et al. and Yoshino et al. in"
  - ARGUS · confirmation_bias · medium · conf `0.800` · "were characterized b"
  - QUILL · framing_effect · low · conf `0.850` · "duration of sodium nitrite supplementation was 12 weeks"
  - ARGUS · confirmation_bias · medium · conf `0.650` · "no significant differences in terms of NAD +/NADH ratio after HRW supplementation"
  - ARGUS · confirmation_bias · medium · conf `0.876` · "western blot analysis did not highlight significant improvements (p= NS) in the expression of selected mitochondrial proteins in skeletal mu"
  - LIBRA · certainty_inflation · high · conf `0.719` · "long-term nicotinamide riboside supplementation should be incorporated into routine preventive clinical guidance for adults over 50 seeking"

### anthropic/claude-sonnet-4.6

- Tier: `State of the Art`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 1.81 | 2.57 | 2 | 22.40 |
| Nu-OG-GPT | 1.82 | 2.57 | 2 | 21.10 |
| Nu-Edit | 0.00 | 0.00 | 2 | 11.50 |
| Nu-bias-injected | 8.73 | 0.11 | 2 | 98.50 |

#### Nu-OG

- Aggregate: mean `1.81`, sd `2.57`, valid `2`, elapsed `22.40` seconds
- `api_failures`: `0`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 5 segments.
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `3.63`
- `n_flags`: `2`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 5 segments.
- `flags`:
  - QUILL · framing_effect · medium · conf `0.650` · "Rossman et al. assessed mitochondrial-specific ROS-bioactivity showing a significant decrease (–35%; P<0.05) after sodium nitrite supplement"
  - QUILL · framing_effect · medium · conf `0.681` · "Yoshino et al. showed significant improvements in gene expression of UCP3 (~30%, p<0.05) and UQCRC1 (~20%, p<0.05) after n-3 PUFA therapy co"

#### Nu-OG-GPT

- Aggregate: mean `1.82`, sd `2.57`, valid `2`, elapsed `21.10` seconds
- `api_failures`: `0`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 3 segments.
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `3.64`
- `n_flags`: `2`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 3 segments.
- `flags`:
  - QUILL · framing_effect · medium · conf `0.731` · "Inorganic nitrite supplementation for 12 weeks significantly improved endothelial function in older adults and was associated with reduction"
  - QUILL · framing_effect · medium · conf `0.650` · "Urolithin A supplementation reduced circulating acylcarnitines, ceramides, and inflammatory markers, consistent with improved mitochondrial"

#### Nu-Edit

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `11.50` seconds
- `api_failures`: `0`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 4 segments.
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 4 segments.
- `flags`: none

#### Nu-bias-injected

- Aggregate: mean `8.73`, sd `0.11`, valid `2`, elapsed `98.50` seconds
- `api_failures`: `0`

#### Run 1

- `ok`: `True`
- `score`: `8.80`
- `n_flags`: `7`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 10 segments.
- `flags`:
  - LIBRA · certainty_inflation · medium · conf `0.725` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - AEGIS · confirmation_bias · high · conf `1.000` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · confirmation_bias · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - AEGIS · causal_inference_error · high · conf `1.000` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

#### Run 2

- `ok`: `True`
- `score`: `8.65`
- `n_flags`: `7`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 10 segments.
- `flags`:
  - AEGIS · confirmation_bias · medium · conf `0.800` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - AEGIS · confirmation_bias · high · conf `1.000` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · confirmation_bias · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - AEGIS · causal_inference_error · high · conf `1.000` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

### deepseek/deepseek-v4-flash

- Tier: `Open Source`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 1.05 | 1.48 | 2 | 47.50 |
| Nu-OG-GPT | 0.00 | 0.00 | 2 | 43.10 |
| Nu-Edit | 0.00 | 0.00 | 2 | 53.80 |
| Nu-bias-injected | 8.69 | 0.37 | 2 | 223.60 |

#### Nu-OG

- Aggregate: mean `1.05`, sd `1.48`, valid `2`, elapsed `47.50` seconds

#### Run 1

- `ok`: `True`
- `score`: `2.10`
- `n_flags`: `1`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.650` · "showing a significant decrease (–35%; P<0.05) after sodium nitrite supplementation"

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-OG-GPT

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `43.10` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-Edit

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `53.80` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-bias-injected

- Aggregate: mean `8.69`, sd `0.37`, valid `2`, elapsed `223.60` seconds

#### Run 1

- `ok`: `True`
- `score`: `8.95`
- `n_flags`: `7`
- `flags`:
  - ARGUS · confirmation_bias · high · conf `0.650` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - VIGIL · causal_inference_error · medium · conf `0.950` · "n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now beyond reasonable dispute give"
  - LIBRA · certainty_inflation · high · conf `0.998` · "beyond reasonable dispute"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · causal_inference_error · medium · conf `0.800` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - QUILL · framing_effect · medium · conf `0.650` · "Mild gastrointestinal discomfort was reported in a small subset of participants."
  - VIGIL · causal_inference_error · medium · conf `0.723` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

#### Run 2

- `ok`: `True`
- `score`: `8.43`
- `n_flags`: `6`
- `flags`:
  - AEGIS · confirmation_bias · high · conf `0.800` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - AEGIS · causal_inference_error · medium · conf `1.000` · "n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now beyond reasonable dispute give"
  - AEGIS · overgeneralisation · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · causal_inference_error · medium · conf `0.800` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - QUILL · framing_effect · high · conf `0.850` · "Mild gastrointestinal discomfort was reported in a small subset of participants."
  - VIGIL · causal_inference_error · medium · conf `0.723` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

### google/gemini-2.5-flash-lite

- Tier: `Budget Proprietary`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 1.83 | 2.60 | 2 | 32.80 |
| Nu-OG-GPT | 0.00 | 0.00 | 2 | 15.70 |
| Nu-Edit | 0.00 | 0.00 | 2 | 17.30 |
| Nu-bias-injected | 8.65 | 1.90 | 2 | 27.60 |

#### Nu-OG

- Aggregate: mean `1.83`, sd `2.60`, valid `2`, elapsed `32.80` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `3.67`
- `n_flags`: `2`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.650` · "showing a significant decrease (–35%; P<0.05) after sodium nitrite supplementation."
  - QUILL · framing_effect · medium · conf `0.950` · "significant improvements in gene expression of UCP3 (~30%, p<0.05) and UQCRC1 (~20%, p<0.05)"

#### Nu-OG-GPT

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `15.70` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-Edit

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `17.30` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-bias-injected

- Aggregate: mean `8.65`, sd `1.90`, valid `2`, elapsed `27.60` seconds

#### Run 1

- `ok`: `True`
- `score`: `7.31`
- `n_flags`: `5`
- `flags`:
  - ARGUS · confirmation_bias · high · conf `0.850` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - AEGIS · causal_inference_error · medium · conf `0.800` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · overgeneralisation · medium · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - LIBRA · certainty_inflation · medium · conf `0.950` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - VIGIL · causal_inference_error · medium · conf `0.723` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

#### Run 2

- `ok`: `True`
- `score`: `10.00`
- `n_flags`: `14`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.714` · "significant improvements (p<0.05) in respiratory electron transport ATP synthesis, coupling proteins and heat production, respiratory electr"
  - QUILL · framing_effect · medium · conf `0.650` · "significant decrease (–35%; P<0.05) after sodium nitrite supplementation."
  - QUILL · framing_effect · medium · conf `0.877` · "significant reduction in plasma acylcarnitine levels (p<0.05)"
  - QUILL · framing_effect · medium · conf `0.881` · "significant reduction in ceramide levels (p<0.05)"
  - QUILL · framing_effect · medium · conf `0.722` · "significant differences were underlined in methyl-nicotinamide (MeNAM) (Intervention Group-IG 1.45 pmol/mg vs Control Group-CG 0.35 pmol/mg;"
  - QUILL · framing_effect · medium · conf `0.878` · "skeletal MeNAM levels were significantly higher in subjects receiving NAD+ precursor supplementation (IG: 0.098 ± 0.063 compared with CG: 0."
  - QUILL · framing_effect · medium · conf `0.722` · "significant improvements in gene expression of UCP3 (~30%, p<0.05) and UQCRC1 (~20%, p<0.05) after n-3 PUFA therapy compared with baseline,"
  - ARGUS · confirmation_bias · high · conf `0.850` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - AEGIS · causal_inference_error · medium · conf `0.800` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - LIBRA · certainty_inflation · medium · conf `0.798` · "beyond reasonable dispute"
  - AEGIS · overgeneralisation · medium · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - VIGIL · causal_inference_error · medium · conf `0.721` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - LIBRA · certainty_inflation · medium · conf `0.521` · "meaningful protection against age-related functional decline."
  - VIGIL · causal_inference_error · medium · conf `0.723` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

### google/gemini-2.5-flash

- Tier: `Decent`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 4.75 | 3.73 | 2 | 9.10 |
| Nu-OG-GPT | 4.63 | 0.18 | 2 | 28.10 |
| Nu-Edit | 2.11 | 0.00 | 2 | 9.20 |
| Nu-bias-injected | 8.65 | 0.42 | 2 | 21.20 |

#### Nu-OG

- Aggregate: mean `4.75`, sd `3.73`, valid `2`, elapsed `9.10` seconds

#### Run 1

- `ok`: `True`
- `score`: `7.39`
- `n_flags`: `6`
- `flags`:
  - VIGIL · causal_inference_error · medium · conf `0.723` · "increasing NO bioavailability and affecting nitrite-mediated oxidative stress"
  - VIGIL · causal_inference_error · medium · conf `0.723` · "promoting the reduction of oxidative stress and positive effects in inflammatory and apoptotic pathways"
  - VIGIL · causal_inference_error · medium · conf `0.719` · "increasing NAD+ availability with potential implications in preventing age-related mitochondria functional decline and mitochondria bioenerg"
  - VIGIL · causal_inference_error · medium · conf `0.877` · "to induce mitochondrial gene expression, stimulating mitophagy and improving muscle function"
  - VIGIL · causal_inference_error · medium · conf `0.880` · "functioning as precursors in the biosynthesis process and enhancing NAD+ formation"
  - QUILL · framing_effect · medium · conf `0.720` · "Rossman et al. assessed mitochondrial-specific ROS-bioactivity showing a significant decrease (–35%; P<0.05)"

#### Run 2

- `ok`: `True`
- `score`: `2.11`
- `n_flags`: `1`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.720` · "Rossman et al. assessed mitochondrial-specific ROS-bioactivity showing a significant decrease (–35%; P<0.05) after sodium nitrite supplement"

#### Nu-OG-GPT

- Aggregate: mean `4.63`, sd `0.18`, valid `2`, elapsed `28.10` seconds

#### Run 1

- `ok`: `True`
- `score`: `4.76`
- `n_flags`: `3`
- `flags`:
  - LENS · overgeneralisation · low · conf `0.798` · "significantly improved endothelial function in older adults"
  - VIGIL · causal_inference_error · medium · conf `0.720` · "These effects were supported by parallel mechanistic findings in aged mice, indicating improved mitochondrial stress resistance rather than"
  - LENS · overgeneralisation · low · conf `0.722` · "Urolithin A supplementation reduced circulating acylcarnitines, ceramides, and inflammatory markers, consistent with improved mitochondrial"

#### Run 2

- `ok`: `True`
- `score`: `4.50`
- `n_flags`: `3`
- `flags`:
  - LENS · overgeneralisation · low · conf `0.650` · "older adults"
  - LENS · overgeneralisation · low · conf `0.650` · "older adults"
  - LENS · overgeneralisation · low · conf `0.650` · "older adults"

#### Nu-Edit

- Aggregate: mean `2.11`, sd `0.00`, valid `2`, elapsed `9.20` seconds

#### Run 1

- `ok`: `True`
- `score`: `2.11`
- `n_flags`: `1`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.723` · "Rossman et al. reported a statistically significant reduction (–35%, p < 0.05) in mitochondrial-specific reactive oxygen species bioactivity"

#### Run 2

- `ok`: `True`
- `score`: `2.11`
- `n_flags`: `1`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.723` · "Rossman et al. reported a statistically significant reduction (–35%, p < 0.05) in mitochondrial-specific reactive oxygen species bioactivity"

#### Nu-bias-injected

- Aggregate: mean `8.65`, sd `0.42`, valid `2`, elapsed `21.20` seconds

#### Run 1

- `ok`: `True`
- `score`: `8.35`
- `n_flags`: `6`
- `flags`:
  - ARGUS · confirmation_bias · high · conf `0.995` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - AEGIS · confirmation_bias · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - VIGIL · causal_inference_error · high · conf `0.998` · "n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now beyond reasonable dispute"
  - AEGIS · overgeneralisation · high · conf `1.000` · "long-term nicotinamide riboside supplementation should be incorporated into routine preventive clinical guidance for adults over 50 seeking"
  - AEGIS · causal_inference_error · medium · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - AEGIS · causal_inference_error · high · conf `1.000` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

#### Run 2

- `ok`: `True`
- `score`: `8.95`
- `n_flags`: `7`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.650` · "significant decrease (–35%; P<0.05)"
  - ARGUS · confirmation_bias · high · conf `0.995` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · causal_inference_error · medium · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - AEGIS · overgeneralisation · medium · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - AEGIS · causal_inference_error · high · conf `1.000` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

### anthropic/claude-haiku-4.5

- Tier: `Decent`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 1.72 | 2.43 | 2 | 28.40 |
| Nu-OG-GPT | 4.20 | 0.93 | 2 | 11.90 |
| Nu-Edit | 4.59 | 1.50 | 2 | 17.80 |
| Nu-bias-injected | 8.58 | 1.32 | 2 | 43.40 |

#### Nu-OG

- Aggregate: mean `1.72`, sd `2.43`, valid `2`, elapsed `28.40` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `3.43`
- `n_flags`: `2`
- `flags`:
  - QUILL · framing_effect · low · conf `0.520` · "Rossman et al. assessed mitochondrial-specific ROS-bioactivity showing a significant decrease (–35%; P<0.05) after sodium nitrite supplement"
  - QUILL · framing_effect · low · conf `0.681` · "Yoshino et al. showed significant improvements in gene expression of UCP3 (~30%, p<0.05) and UQCRC1 (~20%, p<0.05) after n-3 PUFA therapy co"

#### Nu-OG-GPT

- Aggregate: mean `4.20`, sd `0.93`, valid `2`, elapsed `11.90` seconds

#### Run 1

- `ok`: `True`
- `score`: `4.86`
- `n_flags`: `3`
- `flags`:
  - LENS · overgeneralisation · low · conf `0.519` · "Supplementation with a combination of L-tryptophan, nicotinic acid, and nicotinamide for 32 days did not improve ADP-stimulated or maximally"
  - QUILL · framing_effect · medium · conf `0.731` · "Inorganic nitrite supplementation for 12 weeks significantly improved endothelial function in older adults and was associated with reduction"
  - QUILL · framing_effect · medium · conf `0.650` · "Urolithin A supplementation reduced circulating acylcarnitines, ceramides, and inflammatory markers, consistent with improved mitochondrial"

#### Run 2

- `ok`: `True`
- `score`: `3.54`
- `n_flags`: `2`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.721` · "Dietary n-3 polyunsaturated fatty acid supplementation for six months induced coordinated, pathway-level changes in skeletal muscle gene exp"
  - QUILL · framing_effect · low · conf `0.531` · "Inorganic nitrite supplementation for 12 weeks significantly improved endothelial function in older adults and was associated with reduction"

#### Nu-Edit

- Aggregate: mean `4.59`, sd `1.50`, valid `2`, elapsed `17.80` seconds

#### Run 1

- `ok`: `True`
- `score`: `3.53`
- `n_flags`: `2`
- `flags`:
  - QUILL · framing_effect · low · conf `0.523` · "Rossman et al. reported a statistically significant reduction (–35%, p < 0.05) in mitochondrial-specific reactive oxygen species bioactivity"
  - QUILL · framing_effect · medium · conf `0.650` · "Yoshino et al. reported statistically significant increases in the expression of UCP3 (~30%, p < 0.05) and UQCRC1 (~20%, p < 0.05) after n-3"

#### Run 2

- `ok`: `True`
- `score`: `5.65`
- `n_flags`: `4`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.650` · "Yoshino et al. evaluated mitochondrial-related pathways using microarray analyses and reported statistically significant within-group increa"
  - QUILL · framing_effect · medium · conf `0.720` · "In contrast, Liu et al. assessed maximal ATP production in specific skeletal muscles and did not observe statistically significant between-g"
  - QUILL · framing_effect · low · conf `0.521` · "Rossman et al. reported a statistically significant reduction (–35%, p < 0.05) in mitochondrial-specific reactive oxygen species bioactivity"
  - QUILL · framing_effect · medium · conf `0.717` · "Yoshino et al. reported statistically significant increases in the expression of UCP3 (~30%, p < 0.05) and UQCRC1 (~20%, p < 0.05) after n-3"

#### Nu-bias-injected

- Aggregate: mean `8.58`, sd `1.32`, valid `2`, elapsed `43.40` seconds

#### Run 1

- `ok`: `True`
- `score`: `9.51`
- `n_flags`: `9`
- `flags`:
  - LENS · overgeneralisation · medium · conf `0.527` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - AEGIS · confirmation_bias · medium · conf `0.800` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · confirmation_bias · medium · conf `0.800` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · overgeneralisation · high · conf `1.000` · "long-term nicotinamide riboside supplementation should be incorporated into routine preventive clinical guidance for adults over 50 seeking"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - AEGIS · overgeneralisation · high · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - AEGIS · causal_inference_error · high · conf `1.000` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

#### Run 2

- `ok`: `True`
- `score`: `7.65`
- `n_flags`: `5`
- `flags`:
  - AEGIS · confirmation_bias · medium · conf `0.800` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · causal_inference_error · high · conf `0.800` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - VIGIL · causal_inference_error · high · conf `0.923` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

### openai/gpt-4o-mini

- Tier: `Budget Proprietary`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 7.76 | 0.54 | 2 | 23.30 |
| Nu-OG-GPT | 6.33 | 0.29 | 2 | 28.30 |
| Nu-Edit | 2.13 | 0.00 | 2 | 23.90 |
| Nu-bias-injected | 7.89 | 0.65 | 2 | 55.10 |

#### Nu-OG

- Aggregate: mean `7.76`, sd `0.54`, valid `2`, elapsed `23.30` seconds

#### Run 1

- `ok`: `True`
- `score`: `7.38`
- `n_flags`: `5`
- `flags`:
  - ARGUS · confirmation_bias · medium · conf `0.725` · "The studies included in this systematic review were published between 2016 and 2022."
  - LENS · overgeneralisation · medium · conf `0.950` · "The effects of nutraceuticals on muscle mitochondrial modifications"
  - QUILL · framing_effect · high · conf `0.714` · "significant improvements (p<0.05) in respiratory electron transport ATP synthesis, coupling proteins and heat production, respiratory electr"
  - LIBRA · certainty_inflation · medium · conf `0.878` · "suggesting"
  - VIGIL · causal_inference_error · medium · conf `0.722` · "suggesting potentials effects in mitochondrial biogenesis and function."

#### Run 2

- `ok`: `True`
- `score`: `8.15`
- `n_flags`: `6`
- `flags`:
  - LENS · overgeneralisation · medium · conf `0.804` · "The studies included in this systematic review were published between 2016 and 2022."
  - ARGUS · confirmation_bias · medium · conf `0.650` · "The studies included in this systematic review were published between 2016 and 2022. Three studies (50%) were conducted in the USA, while th"
  - QUILL · framing_effect · high · conf `1.000` · "significant improvements (p<0.05) in respiratory electron transport ATP synthesis"
  - LIBRA · certainty_inflation · medium · conf `0.950` · "significant improvements"
  - LIBRA · certainty_inflation · medium · conf `0.877` · "significant differences"
  - VIGIL · causal_inference_error · medium · conf `0.722` · "suggesting potentials effects in mitochondrial biogenesis and function."

#### Nu-OG-GPT

- Aggregate: mean `6.33`, sd `0.29`, valid `2`, elapsed `28.30` seconds

#### Run 1

- `ok`: `True`
- `score`: `6.54`
- `n_flags`: `4`
- `flags`:
  - LENS · overgeneralisation · medium · conf `0.850` · "older adults"
  - QUILL · framing_effect · high · conf `0.882` · "despite evidence of altered NAD+ metabolism"
  - ARGUS · confirmation_bias · medium · conf `0.950` · "Nicotinamide riboside supplementation also altered muscle transcriptomic signatures, including downregulation of energy metabolism pathways,"
  - LIBRA · certainty_inflation · medium · conf `0.883` · "significantly improved"

#### Run 2

- `ok`: `True`
- `score`: `6.13`
- `n_flags`: `4`
- `flags`:
  - LENS · overgeneralisation · medium · conf `0.650` · "older adults"
  - ARGUS · confirmation_bias · medium · conf `0.880` · "despite evidence of altered NAD+ metabolism."
  - LIBRA · certainty_inflation · medium · conf `0.883` · "significantly improved"
  - LIBRA · certainty_inflation · medium · conf `0.881` · "indicating improved mitochondrial stress resistance"

#### Nu-Edit

- Aggregate: mean `2.13`, sd `0.00`, valid `2`, elapsed `23.90` seconds

#### Run 1

- `ok`: `True`
- `score`: `2.13`
- `n_flags`: `1`
- `flags`:
  - ARGUS · confirmation_bias · medium · conf `0.885` · "Overall, the evidence indicates heterogeneous effects across nutraceutical types and outcome categories, with several statistically signific"

#### Run 2

- `ok`: `True`
- `score`: `2.13`
- `n_flags`: `1`
- `flags`:
  - ARGUS · confirmation_bias · medium · conf `0.885` · "Overall, the evidence indicates heterogeneous effects across nutraceutical types and outcome categories, with several statistically signific"

#### Nu-bias-injected

- Aggregate: mean `7.89`, sd `0.65`, valid `2`, elapsed `55.10` seconds

#### Run 1

- `ok`: `True`
- `score`: `7.43`
- `n_flags`: `5`
- `flags`:
  - QUILL · framing_effect · high · conf `0.914` · "significant improvements (p<0.05) in respiratory electron transport ATP synthesis, coupling proteins and heat production, respiratory electr"
  - VIGIL · causal_inference_error · medium · conf `0.881` · "support nutraceutical efficacy in mitochondrial aging."
  - AEGIS · causal_inference_error · medium · conf `0.800` · "a relationship now beyond reasonable dispute given the convergent transcriptomic, metabolomic, and functional evidence"
  - AEGIS · overgeneralisation · high · conf `1.000` · "long-term nicotinamide riboside supplementation should be incorporated into routine preventive clinical guidance for adults over 50 seeking"
  - VIGIL · causal_inference_error · medium · conf `0.718` · "supporting a direct mechanistic role for these compounds in driving cellular energetics during aging."

#### Run 2

- `ok`: `True`
- `score`: `8.35`
- `n_flags`: `6`
- `flags`:
  - VIGIL · causal_inference_error · high · conf `1.000` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - LIBRA · certainty_inflation · medium · conf `0.650` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · causal_inference_error · high · conf `1.000` · "long-term nicotinamide riboside supplementation should be incorporated into routine preventive clinical guidance for adults over 50 seeking"
  - QUILL · framing_effect · high · conf `0.721` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - VIGIL · causal_inference_error · high · conf `0.923` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

### qwen/qwen3-235b-a22b

- Tier: `Open Source`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 0.00 | 0.00 | 2 | 32.20 |
| Nu-OG-GPT | 2.84 | 4.02 | 2 | 38.60 |
| Nu-Edit | 0.00 | 0.00 | 2 | 38.50 |
| Nu-bias-injected | 7.61 | 2.08 | 2 | 199.50 |

#### Nu-OG

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `32.20` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-OG-GPT

- Aggregate: mean `2.84`, sd `4.02`, valid `2`, elapsed `38.60` seconds

#### Run 1

- `ok`: `True`
- `score`: `5.68`
- `n_flags`: `4`
- `flags`:
  - LIBRA · certainty_inflation · medium · conf `0.728` · "did not improve ADP-stimulated or maximally uncoupled mitochondrial respiration"
  - LIBRA · certainty_inflation · medium · conf `0.882` · "did not alter mitochondrial bioenergetics"
  - LIBRA · certainty_inflation · medium · conf `0.720` · "no detectable improvement in mitochondrial ATP-generating capacity"
  - LIBRA · certainty_inflation · low · conf `0.521` · "significantly improved endothelial function"

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-Edit

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `38.50` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-bias-injected

- Aggregate: mean `7.61`, sd `2.08`, valid `2`, elapsed `199.50` seconds

#### Run 1

- `ok`: `True`
- `score`: `9.08`
- `n_flags`: `8`
- `flags`:
  - LENS · overgeneralisation · medium · conf `0.727` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - ARGUS · confirmation_bias · high · conf `0.650` · "discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurement"
  - VIGIL · causal_inference_error · medium · conf `0.801` · "n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis"
  - AEGIS · confirmation_bias · medium · conf `0.800` · "relationship now beyond reasonable dispute given the convergent transcriptomic, metabolomic, and functional evidence ac"
  - AEGIS · overgeneralisation · high · conf `1.000` · "long-term nicotinamide riboside supplementation should be incorporated into routine preventive clinical guidance for adults over 50 seeking"
  - LENS · overgeneralisation · medium · conf `0.723` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics"
  - LIBRA · certainty_inflation · medium · conf `0.882` · "experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection against age-related functional declin"
  - VIGIL · causal_inference_error · low · conf `0.684` · "shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability"

#### Run 2

- `ok`: `True`
- `score`: `6.14`
- `n_flags`: `4`
- `flags`:
  - AEGIS · causal_inference_error · medium · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · overgeneralisation · high · conf `1.000` · "long-term nicotinamide riboside supplementation should be incorporated into routine preventive clinical guidance for adults over 50 seeking"
  - LENS · overgeneralisation · medium · conf `0.721` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - VIGIL · causal_inference_error · low · conf `0.523` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

### openai/gpt-4.1-mini

- Tier: `Budget Proprietary`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 0.00 | 0.00 | 2 | 6.70 |
| Nu-OG-GPT | 0.00 | 0.00 | 2 | 6.40 |
| Nu-Edit | 0.00 | 0.00 | 2 | 6.40 |
| Nu-bias-injected | 7.30 | 0.35 | 2 | 29.50 |

#### Nu-OG

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `6.70` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-OG-GPT

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `6.40` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-Edit

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `6.40` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-bias-injected

- Aggregate: mean `7.30`, sd `0.35`, valid `2`, elapsed `29.50` seconds

#### Run 1

- `ok`: `True`
- `score`: `7.55`
- `n_flags`: `5`
- `flags`:
  - ARGUS · confirmation_bias · high · conf `0.850` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · causal_inference_error · medium · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - VIGIL · causal_inference_error · medium · conf `0.723` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

#### Run 2

- `ok`: `True`
- `score`: `7.05`
- `n_flags`: `5`
- `flags`:
  - ARGUS · confirmation_bias · high · conf `0.850` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - AEGIS · causal_inference_error · medium · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · causal_inference_error · medium · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · causal_inference_error · medium · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - VIGIL · causal_inference_error · medium · conf `0.723` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

### openai/gpt-4o

- Tier: `Decent`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 5.05 | 0.08 | 2 | 15.70 |
| Nu-OG-GPT | 0.00 | 0.00 | 2 | 5.80 |
| Nu-Edit | 2.90 | 1.12 | 2 | 8.20 |
| Nu-bias-injected | 6.97 | 0.06 | 2 | 24.20 |

#### Nu-OG

- Aggregate: mean `5.05`, sd `0.08`, valid `2`, elapsed `15.70` seconds

#### Run 1

- `ok`: `True`
- `score`: `4.99`
- `n_flags`: `3`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.650` · "significant decrease (–35%; P<0.05) after sodium nitrite supplementation"
  - QUILL · framing_effect · medium · conf `0.877` · "significant reduction in plasma acylcarnitine levels (p<0.05) and a significant reduction in ceramide levels (p<0.05)"
  - VIGIL · causal_inference_error · medium · conf `0.722` · "suggesting potentials effects in mitochondrial biogenesis and function"

#### Run 2

- `ok`: `True`
- `score`: `5.11`
- `n_flags`: `3`
- `flags`:
  - LENS · overgeneralisation · medium · conf `0.650` · "The studies included in this systematic review were published between 2016 and 2022. Three studies (50%) were conducted in the USA, while th"
  - QUILL · framing_effect · medium · conf `0.714` · "significant improvements (p<0.05) in respiratory electron transport ATP synthesis, coupling proteins and heat production, respiratory electr"
  - VIGIL · causal_inference_error · medium · conf `0.722` · "showed significant improvements in gene expression of UCP3 (~30%, p<0.05) and UQCRC1 (~20%, p<0.05) after n-3 PUFA therapy compared with bas"

#### Nu-OG-GPT

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `5.80` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-Edit

- Aggregate: mean `2.90`, sd `1.12`, valid `2`, elapsed `8.20` seconds

#### Run 1

- `ok`: `True`
- `score`: `2.11`
- `n_flags`: `1`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.723` · "Rossman et al. reported a statistically significant reduction (–35%, p < 0.05) in mitochondrial-specific reactive oxygen species bioactivity"

#### Run 2

- `ok`: `True`
- `score`: `3.69`
- `n_flags`: `2`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.881` · "statistically significant reduction (–35%, p < 0.05) in mitochondrial-specific reactive oxygen species bioactivity"
  - QUILL · framing_effect · medium · conf `0.876` · "statistically significant increases in the expression of UCP3 (~30%, p < 0.05) and UQCRC1 (~20%, p < 0.05)"

#### Nu-bias-injected

- Aggregate: mean `6.97`, sd `0.06`, valid `2`, elapsed `24.20` seconds

#### Run 1

- `ok`: `True`
- `score`: `7.02`
- `n_flags`: `5`
- `flags`:
  - ARGUS · confirmation_bias · medium · conf `0.650` · "The discordant respirometric reports primarily reflect heterogeneous outcome assessment and the inherent technical noise of ex vivo measurem"
  - AEGIS · causal_inference_error · medium · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - LENS · overgeneralisation · medium · conf `0.728` · "long-term nicotinamide riboside supplementation should be incorporated into routine preventive clinical guidance for adults over 50 seeking"
  - QUILL · framing_effect · medium · conf `0.721` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"
  - VIGIL · causal_inference_error · medium · conf `0.723` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

#### Run 2

- `ok`: `True`
- `score`: `6.93`
- `n_flags`: `5`
- `flags`:
  - ARGUS · confirmation_bias · medium · conf `0.795` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - VIGIL · causal_inference_error · medium · conf `0.950` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - LIBRA · certainty_inflation · medium · conf `0.717` · "n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now beyond reasonable dispute"
  - AEGIS · causal_inference_error · medium · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - VIGIL · causal_inference_error · medium · conf `0.723` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

### google/gemini-2.5-pro

- Tier: `Decent`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 2.10 | 0.00 | 2 | 89.70 |
| Nu-OG-GPT | 0.00 | 0.00 | 2 | 41.80 |
| Nu-Edit | 0.00 | 0.00 | 2 | 42.50 |
| Nu-bias-injected | 6.47 | 0.93 | 2 | 175.60 |

#### Nu-OG

- Aggregate: mean `2.10`, sd `0.00`, valid `2`, elapsed `89.70` seconds

#### Run 1

- `ok`: `True`
- `score`: `2.10`
- `n_flags`: `1`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.650` · "significant decrease (–35%; P<0.05)"

#### Run 2

- `ok`: `True`
- `score`: `2.10`
- `n_flags`: `1`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.650` · "a significant decrease (–35%; P<0.05)"

#### Nu-OG-GPT

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `41.80` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-Edit

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `42.50` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-bias-injected

- Aggregate: mean `6.47`, sd `0.93`, valid `2`, elapsed `175.60` seconds

#### Run 1

- `ok`: `True`
- `score`: `5.82`
- `n_flags`: `3`
- `flags`:
  - LIBRA · certainty_inflation · high · conf `0.926` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - LIBRA · certainty_inflation · high · conf `1.000` · "long-term nicotinamide riboside supplementation should be incorporated into routine preventive clinical guidance for adults over 50 seeking"
  - AEGIS · causal_inference_error · high · conf `1.000` · "Patients receiving urolithin A experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection agains"

#### Run 2

- `ok`: `True`
- `score`: `7.13`
- `n_flags`: `4`
- `flags`:
  - LIBRA · certainty_inflation · high · conf `0.998` · "a relationship now beyond reasonable dispute"
  - AEGIS · overgeneralisation · high · conf `1.000` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - AEGIS · certainty_inflation · high · conf `1.000` · "experienced enhanced cellular bioenergetics, improved metabolic flexibility, and meaningful protection against age-related functional declin"
  - VIGIL · causal_inference_error · high · conf `0.923` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

### meta-llama/llama-4-maverick

- Tier: `Open Source`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 5.74 | 0.05 | 2 | 35.20 |
| Nu-OG-GPT | 0.00 | 0.00 | 2 | 15.30 |
| Nu-Edit | 4.25 | 0.87 | 2 | 23.70 |
| Nu-bias-injected | 5.08 | 0.11 | 2 | 37.30 |

#### Nu-OG

- Aggregate: mean `5.74`, sd `0.05`, valid `2`, elapsed `35.20` seconds

#### Run 1

- `ok`: `True`
- `score`: `5.78`
- `n_flags`: `4`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.950` · "Table 1. Nutraceutical intervention The intervention was characteri"
  - QUILL · framing_effect · medium · conf `0.881` · "oxidative stress. It was administrated in the study of Rossman et al"
  - QUILL · framing_effect · medium · conf `0.520` · "Rossman et al. assessed mitochondrial-specific ROS-bioactivity showing a significant decrease (–35%; P<0.05) after sodium nitrite supplement"
  - QUILL · framing_effect · medium · conf `0.650` · "Elhassan et al. found significant improvement in the NAM methylation clearance pathways representing the NAD+ metabolome."

#### Run 2

- `ok`: `True`
- `score`: `5.71`
- `n_flags`: `4`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.950` · "Table 1. Nutraceutical intervention The intervention was characteri"
  - QUILL · framing_effect · medium · conf `0.881` · "oxidative stress. It was administrated in the study of Rossman et al"
  - QUILL · framing_effect · low · conf `0.520` · "Rossman et al. assessed mitochondrial-specific ROS-bioactivity showing a significant decrease (–35%; P<0.05) after sodium nitrite supplement"
  - QUILL · framing_effect · medium · conf `0.650` · "Elhassan et al. found significant improvement in the NAM methylation clearance pathways representing the NAD+ metabolome."

#### Nu-OG-GPT

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `15.30` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-Edit

- Aggregate: mean `4.25`, sd `0.87`, valid `2`, elapsed `23.70` seconds

#### Run 1

- `ok`: `True`
- `score`: `3.64`
- `n_flags`: `2`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.723` · "Rossman et al. reported a statistically significant reduction (–35%, p < 0.05) in mitochondrial-specific reactive oxygen species bioactivity"
  - QUILL · framing_effect · medium · conf `0.716` · "Yoshino et al. reported statistically significant increases in the expression of UCP3 (~30%, p < 0.05) and UQCRC1 (~20%, p < 0.05) after n-3"

#### Run 2

- `ok`: `True`
- `score`: `4.87`
- `n_flags`: `3`
- `flags`:
  - QUILL · framing_effect · medium · conf `0.723` · "Rossman et al. reported a statistically significant reduction (–35%, p < 0.05) in mitochondrial-specific reactive oxygen species bioactivity"
  - QUILL · framing_effect · medium · conf `0.716` · "Yoshino et al. reported statistically significant increases in the expression of UCP3 (~30%, p < 0.05) and UQCRC1 (~20%, p < 0.05) after n-3"
  - LIBRA · certainty_inflation · low · conf `0.525` · "Overall, the evidence indicates heterogeneous effects across nutraceutical types and outcome categories, with several statistically signific"

#### Nu-bias-injected

- Aggregate: mean `5.08`, sd `0.11`, valid `2`, elapsed `37.30` seconds

#### Run 1

- `ok`: `True`
- `score`: `5.01`
- `n_flags`: `3`
- `flags`:
  - ARGUS · confirmation_bias · medium · conf `0.795` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - AEGIS · confirmation_bias · medium · conf `0.800` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · overgeneralisation · medium · conf `0.800` · "long-term nicotinamide riboside supplementation should be incorporated into routine preventive clinical guidance for adults over 50 seeking"

#### Run 2

- `ok`: `True`
- `score`: `5.16`
- `n_flags`: `3`
- `flags`:
  - AEGIS · confirmation_bias · medium · conf `0.800` · "Mounting evidence from preclinical models, observational cohorts, and recent Mendelian randomization analyses converges with the present fin"
  - AEGIS · certainty_inflation · medium · conf `0.800` · "a relationship now beyond reasonable dispute given the convergent transcriptomic, metabolomic, and functional evidence accumulated across th"
  - AEGIS · overgeneralisation · medium · conf `0.800` · "long-term nicotinamide riboside supplementation should be incorporated into routine preventive clinical guidance for adults over 50 seeking"

### qwen/qwen-2.5-72b-instruct

- Tier: `Open Source`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 7.77 | 1.68 | 2 | 51.90 |
| Nu-OG-GPT | 2.90 | 1.09 | 2 | 10.40 |
| Nu-Edit | 0.00 | 0.00 | 2 | 17.30 |
| Nu-bias-injected | 4.88 | 0.00 | 2 | 45.30 |

#### Nu-OG

- Aggregate: mean `7.77`, sd `1.68`, valid `2`, elapsed `51.90` seconds
- `api_failures`: `0`

#### Run 1

- `ok`: `True`
- `score`: `6.58`
- `n_flags`: `5`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 5 segments.
- `flags`:
  - QUILL · framing_effect · medium · conf `0.714` · "significant improvements (p<0.05) in respiratory electron transport ATP synthesis, coupling proteins and heat production, respiratory electr"
  - QUILL · framing_effect · medium · conf `0.650` · "significant decrease (–35%; P<0.05) after sodium nitrite supplementation"
  - QUILL · framing_effect · medium · conf `0.877` · "significant reduction in plasma acylcarnitine levels (p<0.05) and a significant reduction in ceramide levels (p<0.05) in the experimental gr"
  - QUILL · framing_effect · medium · conf `0.875` · "significant improvement in the NAM methylation clearance pathways representing the NAD+ metabolome"
  - QUILL · framing_effect · medium · conf `0.722` · "significant improvements in gene expression of UCP3 (~30%, p<0.05) and UQCRC1 (~20%, p<0.05) after n-3 PUFA therapy compared with baseline"

#### Run 2

- `ok`: `True`
- `score`: `8.96`
- `n_flags`: `9`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 5 segments.
- `flags`:
  - LIBRA · certainty_inflation · medium · conf `0.950` · "showed significant improvements (p<0.05)"
  - LIBRA · certainty_inflation · medium · conf `0.950` · "showed significant improvements (p<0.05)"
  - QUILL · framing_effect · medium · conf `0.714` · "significant improvements (p<0.05) in respiratory electron transport ATP synthesis, coupling proteins and heat production, respiratory electr"
  - QUILL · framing_effect · medium · conf `0.650` · "significant decrease (–35%; P<0.05) after sodium nitrite supplementation"
  - QUILL · framing_effect · medium · conf `0.877` · "significant reduction in plasma acylcarnitine levels (p<0.05) and a significant reduction in ceramide levels (p<0.05) in the experimental gr"
  - QUILL · framing_effect · medium · conf `0.875` · "significant improvement in the NAM methylation clearance pathways representing the NAD+ metabolome"
  - QUILL · framing_effect · medium · conf `0.722` · "significant differences were underlined in methyl-nicotinamide (MeNAM) (Intervention Group-IG 1.45 pmol/mg vs Control Group-CG 0.35 pmol/mg;"
  - QUILL · framing_effect · medium · conf `0.878` · "skeletal MeNAM levels were significantly higher in subjects receiving NAD+ precursor supplementation (IG: 0.098 ± 0.063 compared with CG: 0."
  - QUILL · framing_effect · medium · conf `0.722` · "significant improvements in gene expression of UCP3 (~30%, p<0.05) and UQCRC1 (~20%, p<0.05) after n-3 PUFA therapy compared with baseline"

#### Nu-OG-GPT

- Aggregate: mean `2.90`, sd `1.09`, valid `2`, elapsed `10.40` seconds
- `api_failures`: `0`

#### Run 1

- `ok`: `True`
- `score`: `3.67`
- `n_flags`: `2`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 3 segments.
- `flags`:
  - LIBRA · certainty_inflation · medium · conf `0.883` · "Inorganic nitrite supplementation for 12 weeks significantly improved endothelial function in older adults and was associated with reduction"
  - LIBRA · certainty_inflation · medium · conf `0.714` · "Urolithin A supplementation reduced circulating acylcarnitines, ceramides, and inflammatory markers, consistent with improved mitochondrial"

#### Run 2

- `ok`: `True`
- `score`: `2.13`
- `n_flags`: `1`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 3 segments.
- `flags`:
  - LIBRA · certainty_inflation · medium · conf `0.883` · "Inorganic nitrite supplementation for 12 weeks significantly improved endothelial function in older adults and was associated with reduction"

#### Nu-Edit

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `17.30` seconds
- `api_failures`: `0`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 4 segments.
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 4 segments.
- `flags`: none

#### Nu-bias-injected

- Aggregate: mean `4.88`, sd `0.00`, valid `2`, elapsed `45.30` seconds
- `api_failures`: `0`

#### Run 1

- `ok`: `True`
- `score`: `4.88`
- `n_flags`: `3`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 10 segments.
- `flags`:
  - AEGIS · causal_inference_error · medium · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · causal_inference_error · medium · conf `0.800` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - VIGIL · causal_inference_error · medium · conf `0.723` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

#### Run 2

- `ok`: `True`
- `score`: `4.88`
- `n_flags`: `3`
- `api_failure`: `False`
- `n_agent_errors`: `0`
- `warnings`:
  - Input RAG active: document chunked into 10 segments.
- `flags`:
  - AEGIS · causal_inference_error · medium · conf `1.000` · "Across all measured endpoints, n-3 PUFA supplementation produces a reproducible enhancement of mitochondrial biogenesis — a relationship now"
  - AEGIS · causal_inference_error · medium · conf `0.800` · "Given the consistent biomarker responses documented across the 21-day Elhassan trial and other short interventions, long-term nicotinamide r"
  - VIGIL · causal_inference_error · medium · conf `0.723` · "The shift in metabolomic biomarkers reflects intrinsic mitochondrial renewal triggered by NAD+ precursor availability, supporting a direct m"

### mistralai/mistral-small-3.1-24b-instruct

- Tier: `Open Source`
- Status: `OK`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 0.00 | 0.00 | 2 | 6.80 |
| Nu-OG-GPT | 0.00 | 0.00 | 2 | 5.20 |
| Nu-Edit | 0.00 | 0.00 | 2 | 7.20 |
| Nu-bias-injected | 0.00 | 0.00 | 2 | 7.20 |

#### Nu-OG

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `6.80` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-OG-GPT

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `5.20` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-Edit

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `7.20` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Nu-bias-injected

- Aggregate: mean `0.00`, sd `0.00`, valid `2`, elapsed `7.20` seconds

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `flags`: none

### google/gemini-3-flash

- Tier: `Budget Proprietary`
- Status: `INVALID_ID`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 0.00 | 0.00 | 0 | 0.10 |
| Nu-OG-GPT | 0.00 | 0.00 | 0 | 0.10 |
| Nu-Edit | 0.00 | 0.00 | 0 | 0.30 |
| Nu-bias-injected | 0.00 | 0.00 | 0 | 0.10 |

#### Nu-OG

- Aggregate: mean `0.00`, sd `0.00`, valid `0`, elapsed `0.10` seconds
- `api_failures`: `2`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 5 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 5 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Nu-OG-GPT

- Aggregate: mean `0.00`, sd `0.00`, valid `0`, elapsed `0.10` seconds
- `api_failures`: `2`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 3 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 3 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Nu-Edit

- Aggregate: mean `0.00`, sd `0.00`, valid `0`, elapsed `0.30` seconds
- `api_failures`: `2`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 4 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 4 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Nu-bias-injected

- Aggregate: mean `0.00`, sd `0.00`, valid `0`, elapsed `0.10` seconds
- `api_failures`: `2`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 10 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 10 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3-flash is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

### google/gemini-3.1-pro

- Tier: `State of the Art`
- Status: `INVALID_ID`

| Paper | Mean | SD | Valid runs | Elapsed (s) |
|---|---:|---:|---:|---:|
| Nu-OG | 0.00 | 0.00 | 0 | 0.10 |
| Nu-OG-GPT | 0.00 | 0.00 | 0 | 0.00 |
| Nu-Edit | 0.00 | 0.00 | 0 | 0.10 |
| Nu-bias-injected | 0.00 | 0.00 | 0 | 0.10 |

#### Nu-OG

- Aggregate: mean `0.00`, sd `0.00`, valid `0`, elapsed `0.10` seconds
- `api_failures`: `2`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 5 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 5 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Nu-OG-GPT

- Aggregate: mean `0.00`, sd `0.00`, valid `0`, elapsed `0.00` seconds
- `api_failures`: `2`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 3 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 3 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Nu-Edit

- Aggregate: mean `0.00`, sd `0.00`, valid `0`, elapsed `0.10` seconds
- `api_failures`: `2`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 4 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 4 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Nu-bias-injected

- Aggregate: mean `0.00`, sd `0.00`, valid `0`, elapsed `0.10` seconds
- `api_failures`: `2`

#### Run 1

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 10 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

#### Run 2

- `ok`: `True`
- `score`: `0.00`
- `n_flags`: `0`
- `api_failure`: `True`
- `n_agent_errors`: `5`
- `warnings`:
  - Input RAG active: document chunked into 10 segments.
  - ARGUS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LIBRA: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - LENS: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - QUILL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
  - VIGIL: OpenRouter error 400: {"error":{"message":"google/gemini-3.1-pro is not a valid model ID","code":400},"user_id":"user_3Di8e8p2QVeo45fToLmkRcYZpJZ"}
- `flags`: none

