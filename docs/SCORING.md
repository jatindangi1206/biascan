# BiasScan — Flags & Scoring

How a piece of text becomes a bias score, end to end. This is the user-facing
contract; the implementation lives in
[`backend/app/agents/orchestrator.py`](../backend/app/agents/orchestrator.py)
(`_overall_score`) and is mirrored in
[`frontend/src/components/ResultsPanel.tsx`](../frontend/src/components/ResultsPanel.tsx).

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
            confidence floor (drop weak flags, < 0.5)
                          │
                          ▼
            AEGIS — conflict resolution    (only if overlapping spans
                                            with different bias types)
                          │
                          ▼
            _overall_score(flags, text) → 0.0 – 1.0
                          │
                          ▼
            scoreValue = score × 10, mapped to a label
```

---

## 1. Flag generation (the five agents)

Five specialised agents run in parallel against the chosen LLM provider. Each
asks the model to find one specific kind of bias and return JSON. Every flag
carries:

| Field | Source | Notes |
|---|---|---|
| `bias_type` | Agent class | Fixed per agent (e.g. ARGUS → `confirmation_bias`) |
| `flagged_text`, `span_start`, `span_end` | LLM | Re-anchored against source if offsets drift |
| `confidence` | LLM | 0.0 – 1.0 |
| `severity` | LLM | `low` / `medium` / `high` — the model's judgment per flag |
| `clean_alternative`, `false_positive_check`, … | LLM | Optional context |

Severity is decided by the LLM per flag; the orchestrator never overrides it.
Flags whose `confidence < CONFIDENCE_FLOOR` (default 0.5, env-overridable) are
dropped before scoring.

---

## 2. AEGIS — conflict resolution

AEGIS only fires when **two or more flags overlap (IoU ≥ 0.5) with different
`bias_type`s** ([`meta_evaluator.py`](../backend/app/agents/meta_evaluator.py)).
Most flags never see it. When it does fire, union-find groups all
transitively-overlapping flags into one cluster, AEGIS reads the source span
plus every candidate flag (and each agent's reasoning) and returns **one**
consolidated annotation per cluster. If AEGIS fails, the highest-confidence
original is kept.

---

## 3. The score formula

The score is a single product of two factors — **how much** of the text is
biased and **how strongly** — calibrated onto a 0–10 scale.

```
coverage = min(1, flagged_words / total_words)
impact   = Σᵢ (confidenceᵢ × severity_weightᵢ)   +   0.15 × (unique_types − 1)
                                                       severity: high=3, medium=2, low=1
score (0–10) = min(10,  impact × coverage × 3.33)
```

Below is each piece, and why it's there.

### Coverage — *how much of the text is biased*

```
coverage = min(1, flagged_words / total_words)
```

**Why.** A bias score has to answer "how much of this document is affected?"
Counting flagged words against total words is the standard length
normalisation: the same two biased sentences should weigh far more in a
50-word abstract than in a 5,000-word report. Each document word is counted
**once** even when several flags overlap it (a union, not a sum), so coverage
is an honest fraction that can never exceed 100% — the `min(1, ·)` is just a
safety clamp. There is no length floor, so a short, fully-biased snippet
legitimately gets `coverage = 1.0`.

### Impact — *how strongly biased it is*

```
impact = Σᵢ confidenceᵢ × severity_weightᵢ
```

**Why.** Two things make a flag matter: how sure the model is (confidence) and
how bad the issue is (severity). Multiplying them per flag and summing gives an
"expected severity" — a flag the model is only 50% sure about counts half.
Summing across flags means more issues raise the score, which is the trust
contract: more flags ⇒ higher score.

### Severity weights — 3 / 2 / 1

```
high = 3.0   medium = 2.0   low = 1.0
```

**Why.** A simple, readable ladder: a high-severity issue counts three times a
low one. Integers keep it auditable — the UI shows `H=3 · M=2 · L=1` so a user
can recompute the number by hand. The exact ratio isn't a psychometric claim;
it's the simplest monotone weighting that preserves ordering.

### Confidence floor — 0.5

Flags below 0.5 confidence are dropped *before* scoring. 0.5 = "more likely
than not." So every flag that reaches the formula contributes between
`0.5 × weight` and `1.0 × weight` — weak guesses are gone, surviving flags are
still discounted by remaining uncertainty.

### Diversity bump — +0.15 per extra bias type

```
impact += 0.15 × max(0, unique_types − 1)
```

**Why.** A document tripping four *different* kinds of bias is more
systemically biased than one tripping the same kind four times. This is a small
additive nudge — at most +0.6 when all five types fire — so it breaks ties
without overpowering coverage or impact. The `−1` means a single-type document
gets nothing extra (that's the baseline), and it's added to `impact` before the
coverage multiply, so it too gets length-scaled.

### Why a product, and why × 3.33

```
score (0–10) = min(10,  impact × coverage × 3.33)
```

**Why a product (not a sum).** A document should score high only if it is both
strongly biased (`impact`) **and** substantially affected (`coverage`). One
severe flag buried in a long clean report has tiny coverage, so the product
stays low — it can't pretend the whole document is biased. That AND-behaviour
is the central design property, and only a product gives it.

**Why 3.33.** It is a calibration constant, ≈ 10/3, that sets where the score
saturates. We want the worst realistic document — fully covered
(`coverage = 1`) with `impact ≈ 3` (e.g. one confident high-severity flag,
`1.0 × 3 = 3`) — to land at the top of the scale, 10. Solve for the constant
`k`:

```
impact × coverage × k = 10
      3   ×    1    × k = 10
                      k = 3.33
```

So 3.33 makes the score climb **fast** — a thoroughly biased passage reaches
the "Severe" band quickly — while `min(10, ·)` caps it so it **never crosses**
the 0–10 range. Without the multiplier you would need `impact × coverage = 10`
to max out, which is practically unreachable, and almost every biased document
would cluster near zero.

The final value is divided by 10 to return a `0.0–1.0` score from the API; the
UI multiplies back by 10 for display.

---

## 4. Score → label

The frontend maps the 0–10 score to one of four bands:

| Score | Label |
|---|---|
| 0.0 – 2.9 | **Low** |
| 3.0 – 5.4 | **Moderate** |
| 5.5 – 7.4 | **Concerning** |
| 7.5 – 10.0 | **Severe** |

---

## 5. Worked examples

**Short biased sentence** — 1 high flag (conf 0.9), whole sentence flagged:

```
coverage = 1.0
impact   = 0.9 × 3                     = 2.70
score    = min(10, 2.70 × 1.0 × 3.33)  = 9.0 / 10  →  "Severe"
```

**Long report, one small flag** — 200 words, one 10-word medium flag (conf 0.7):

```
coverage = 10 / 200                    = 0.05
impact   = 0.7 × 2                     = 1.40
score    = min(10, 1.40 × 0.05 × 3.33) = 0.2 / 10  →  "Low"
```

Same flag, very different score — because coverage says the rest of the
document is clean.

**Moderately biased** — 3 flags over 2 types, ~40% of the text covered
(high @0.8, medium @0.7, medium @0.6):

```
impact   = (0.8×3 + 0.7×2 + 0.6×2) + 0.15×(2−1)  = 5.00 + 0.15 = 5.15
coverage = 0.40
score    = min(10, 5.15 × 0.40 × 3.33)            = 6.9 / 10  →  "Concerning"
```

---

## 6. Keeping it in sync

Two places implement the same formula and must agree:

- [`backend/app/agents/orchestrator.py`](../backend/app/agents/orchestrator.py) — `_overall_score`
- [`frontend/src/components/ResultsPanel.tsx`](../frontend/src/components/ResultsPanel.tsx) — the breakdown under "How is this calculated?"

The frontend recomputes the breakdown locally so users see the same numbers the
score was built from. Change one, change the other.

To re-tune the constants empirically, [`eval/calibrate_score.py`](../eval/calibrate_score.py)
grid-searches scoring constants against the labelled biased/control corpus to
maximise ROC AUC of `biased_score > control_score`.
