# BiasScan — Flags & Scoring

How a piece of text becomes a bias score, end to end. This is the user-facing
contract; the implementation lives in
[`backend/app/agents/orchestrator.py`](../backend/app/agents/orchestrator.py)
and [`frontend/src/components/ResultsPanel.tsx`](../frontend/src/components/ResultsPanel.tsx).

---

## Pipeline overview

```
        ┌── ARGUS  (confirmation_bias)     ──┐
        ├── LIBRA  (certainty_inflation)  ──┤
 text ──┼── LENS   (overgeneralisation)   ──┼── flags (raw)
        ├── QUILL  (framing_effect)        ──┤
        └── VIGIL  (causal_inference)     ──┘
                          │
                          ▼
            confidence floor (drop weak flags)
                          │
                          ▼
            Evidence RAG cross-check  (±0.15 to confidence)
                          │
                          ▼
            AEGIS — conflict resolution    (only if overlapping spans
                                            with different bias types)
                          │
                          ▼
            _overall_score(flags) → 0.0 – 1.0
                          │
                          ▼
            scoreValue = score × 10, mapped to a label
```

Each stage is described below.

---

## 1. Flag generation (the five agents)

Five specialised agents run in parallel against the chosen LLM provider. Each
asks the model to find one specific kind of bias and return JSON. Every
returned flag carries:

| Field | Source | Notes |
|---|---|---|
| `bias_type` | Agent class | Fixed per agent (e.g. ARGUS → `confirmation_bias`) |
| `flagged_text`, `span_start`, `span_end` | LLM | Re-anchored against source if offsets drift |
| `confidence` | LLM | 0.0 – 1.0 |
| `severity` | LLM | `low` / `medium` / `high` — model's judgment per flag |
| `clean_alternative`, `false_positive_check`, … | LLM | Optional context |

**Important:** severity is decided by the LLM per flag. The orchestrator
never overrides it.

Flags whose `confidence < CONFIDENCE_FLOOR` (default 0.5, env-overridable)
are dropped before scoring.

---

## 2. Evidence RAG cross-check

Every surviving flag's `flagged_text` is queried against an evidence index
built from the eleven hand-written `BIAS_PATTERNS` plus dataset exemplars in
`eval/output/samples.json`. Each match adjusts confidence:

| Match label | Effect on confidence |
|---|---|
| `biased` | + up to 0.08 per strong match |
| `neutral` / `control` | − up to 0.08 per strong match |

Total adjustment is clamped to **±0.15** — the cross-check refines, never
overrides, the LLM's judgment.

---

## 3. AEGIS — conflict resolution

AEGIS only fires when **two or more flags overlap (IoU ≥ 0.3) with different
`bias_type`s**. Most flags never see AEGIS. When it does fire:

1. Union-find groups all transitively-overlapping flags into a single cluster.
2. AEGIS reads the source span + every candidate flag (including each agent's
   chain-of-thought) and returns **one** consolidated annotation per cluster.
3. If AEGIS fails, the highest-confidence original is kept as a fallback.

---

## 4. Overall score formula

```
n           = number of surviving flags
base        = 8.0 × (1 − 1 / (1 + n/3))
severity    = min(1.5,  0.4 × high_count  +  0.15 × medium_count)
diversity   = 0.15 × (unique_bias_types − 1)

score (0–10) = min(10, base + severity + diversity)
```

Three properties this guarantees:

1. **Monotonic in flag count.** Adding a flag can never lower the score.
2. **Count dominates.** `base` ranges 0 – 8; severity adds at most +1.5;
   diversity adds at most +0.6 (5 types). Most of the score is "how many
   things were flagged."
3. **Continuous at zero.** No flags → 0.0. One flag never jumps into the
   "Moderate" band.

### Base curve (purely a function of flag count)

| Flags | Base |
|---:|---:|
| 0 | 0.00 |
| 1 | 2.00 |
| 2 | 3.20 |
| 3 | 4.00 |
| 4 | 4.57 |
| 5 | 5.00 |
| 6 | 5.33 |
| 8 | 5.82 |
| 10 | 6.15 |
| 20 | 6.96 |
| ∞ | 8.00 |

### Severity bumps

| Composition | Bump |
|---|---:|
| 1 medium | +0.15 |
| 1 high | +0.40 |
| 2 medium | +0.30 |
| 2 high | +0.80 |
| 3 high | +1.20 |
| 4 high (or many mediums) | **+1.50 (capped)** |

Low-severity flags contribute nothing to this bump.

### Diversity bump

| Distinct bias types | Bump |
|---:|---:|
| 1 | 0.00 |
| 2 | +0.15 |
| 3 | +0.30 |
| 4 | +0.45 |
| 5 | +0.60 |

---

## 5. Score → label

The frontend ([ResultsPanel.tsx](../frontend/src/components/ResultsPanel.tsx))
maps the 0–10 score to one of four bands:

| Score | Label |
|---|---|
| 0.0 – 2.9 | **Low** |
| 3.0 – 5.4 | **Moderate** |
| 5.5 – 7.4 | **Concerning** |
| 7.5 – 10.0 | **Severe** |

---

## 6. Score lookup tables

All values computed by running the live `_overall_score` function on
synthetic input. Use this to sanity-check the UI.

### Pure severity (single bias type)

| Flags | All low | All medium | All high |
|---:|---:|---:|---:|
| 0 | 0.0 | 0.0 | 0.0 |
| 1 | 2.0 | 2.1 | 2.4 |
| 2 | 3.2 | 3.5 | 4.0 |
| 3 | 4.0 | 4.5 | 5.2 |
| 4 | 4.6 | 5.2 | 6.1 |
| 5 | 5.0 | 5.8 | 6.5 |
| 6 | 5.3 | 6.2 | 6.8 |
| 7 | 5.6 | 6.7 | 7.1 |
| 8 | 5.8 | 7.0 | 7.3 |
| 9 | 6.0 | 7.3 | 7.5 |
| 10 | 6.2 | 7.7 | 7.7 |

### Diversity at fixed count (5 medium flags)

| Bias types | Score |
|---:|---:|
| 1 | 5.8 |
| 2 | 5.9 |
| 3 | 6.1 |
| 4 | 6.2 |
| 5 | 6.4 |

---

## 7. Worked examples

### Clean Cochrane Review excerpt — 2 high flags, 2 bias types

```
base       = 8.0 × (1 − 1/(1 + 2/3))   = 3.20
severity   = 0.4 × 2                    = 0.80
diversity  = 0.15 × (2 − 1)             = 0.15
─────────────────────────────────────
score      = 4.15  →  4.2 / 10  →  "Moderate"
```

### Cochrane homeopathy review — 4 flags (2 high + 2 medium), 4 bias types

```
base       = 8.0 × (1 − 1/(1 + 4/3))   = 4.57
severity   = 0.4 × 2 + 0.15 × 2         = 1.10
diversity  = 0.15 × (4 − 1)             = 0.45
─────────────────────────────────────
score      = 6.12  →  6.1 / 10  →  "Concerning"
```

The biased document scores **higher than the clean one even though both
trigger the same model.** This is the trust contract: more flags ⇒ higher
score, every time.

### Single weak finding — 1 medium flag, 1 bias type

```
base       = 8.0 × (1 − 1/(1 + 1/3))   = 2.00
severity   = 0.15
diversity  = 0
─────────────────────────────────────
score      = 2.15  →  2.2 / 10  →  "Low"
```

One flag is one issue. The score reflects that.

---

## 8. Before/after — why this formula replaced the old one

The previous formula used max-pooling + top-k weighted average + a density
multiplier. Two failure modes drove the rewrite:

| Case | Old | New | Why old was wrong |
|---|---:|---:|---|
| 0 flags | 0.0 | 0.0 | — |
| 1 medium flag | **4.0** | 2.2 | Y-intercept of 2.5 made the 0→1 transition a cliff |
| 1 high flag | 4.3 | 2.4 | Same — single flag should not feel "Moderate" |
| 2 high (clean Cochrane) | 5.7 | 4.2 | Density multiplier inflated short docs |
| 4 mixed (homeopathy) | **6.6** | 6.1 | Top-5 weighted *average* let medium flags drag the score *down* — the biased doc with **more** flags scored *lower* than the clean one |
| 3 medium | 5.7 | 4.5 | Same dilution problem |
| 5 high, 5 types | 8.0 | 7.1 | — |

The pathological case — fewer-but-stronger flags beating more-but-mixed
flags — was the immediate bug. Removing the density multiplier and replacing
the average with a count-driven curve fixes it.

---

## 9. Keeping it in sync

Two places implement the same formula and must agree:

- [`backend/app/agents/orchestrator.py`](../backend/app/agents/orchestrator.py) — `_overall_score`
- [`frontend/src/components/ResultsPanel.tsx`](../frontend/src/components/ResultsPanel.tsx) — the breakdown rendered under "How is this calculated?"

If you change the formula in one place, change it in the other. The frontend
recomputes the breakdown locally (rather than relying on the backend to send
it back) so users see the same numbers the score was built from.
