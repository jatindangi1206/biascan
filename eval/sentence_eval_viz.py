"""Build comparison graphs from sentence_eval_per_run.csv.

Outputs PNGs to eval/output/sentence_eval_figs/.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "eval" / "output" / "sentence_eval_per_run.csv"
OUT = ROOT / "eval" / "output" / "sentence_eval_figs"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150

df = pd.read_csv(CSV)
df["any_fire"] = (df["n_flags"] > 0).astype(int)
df["bias_types_detected"] = df["bias_types_detected"].fillna("")
df["agents_fired"] = df["agents_fired"].fillna("")

# Short model labels for plots
SHORT = {
    "openai/gpt-4.1-nano":               "GPT-4.1",
    "anthropic/claude-3-haiku":          "Claude-Haiku-4.5",
    "google/gemini-2.5-flash-lite":      "Gemini-2.5-Flash",
    "deepseek/deepseek-v4-flash":        "DeepSeek-R1",
    "meta-llama/llama-3.3-70b-instruct": "Llama-4-Maverick",
}
df["model_short"] = df["model"].map(SHORT)

BIAS_ORDER = [
    "confirmation_bias",
    "certainty_inflation",
    "framing_effect",
    "causal_inference_error",
    "overgeneralisation",
]
BIAS_LABEL = {
    "confirmation_bias":      "Confirmation",
    "certainty_inflation":    "Certainty",
    "framing_effect":         "Framing",
    "causal_inference_error": "Causal",
    "overgeneralisation":     "Overgen.",
}
df["expected_short"] = df["expected_bias_type"].map(BIAS_LABEL)

MODEL_ORDER = [
    "GPT-4.1", "Claude-Haiku-4.5", "Gemini-2.5-Flash",
    "DeepSeek-R1", "Llama-4-Maverick",
]


# ── 1. Overall accuracy by model ────────────────────────────────────────
# Error bars = ±1 SEM clustered by sentence: the 5 runs of a sentence are
# repeated measures (not independent), so the sentence is the sampling unit.
# Per-sentence accuracy (mean of its 5 runs), then SEM across the 48 sentences.
per_sent = df.groupby(["model_short", "sentence_id"])["correct"].mean() * 100
acc = per_sent.groupby("model_short").mean().reindex(MODEL_ORDER)
sem = (per_sent.groupby("model_short").std(ddof=1)
       / np.sqrt(per_sent.groupby("model_short").count())).reindex(MODEL_ORDER)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(acc.index, acc.values, yerr=sem.values, capsize=5,
              color=sns.color_palette("crest", len(acc)),
              error_kw={"ecolor": "#333333", "elinewidth": 1.2})
for b, v, e in zip(bars, acc.values, sem.values):
    ax.text(b.get_x() + b.get_width()/2, v + e + 1.5, f"{v:.1f}%",
            ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Accuracy (% of runs where expected bias was detected)")
ax.set_title("Overall accuracy by model — 48 sentences × 5 runs each (n=240)")
ax.set_ylim(0, 75)
ax.plot([], [], color="#333333", marker="|", ls="none", markersize=10,
        label="Error bars: ±1 SEM (clustered by sentence, n=48)")
ax.legend()
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT / "01_accuracy_by_model.png")
plt.close()

# ── 2. Accuracy heatmap: model × bias_type ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
mat = df.pivot_table(values="correct", index="model_short",
                     columns="expected_short", aggfunc="mean")
mat = mat.reindex(index=MODEL_ORDER, columns=[BIAS_LABEL[b] for b in BIAS_ORDER]) * 100
sns.heatmap(mat, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100,
            cbar_kws={"label": "Accuracy (%)"}, ax=ax, linewidths=0.5)
ax.set_title("Accuracy heatmap — model × expected bias type (%)")
ax.set_xlabel("Expected bias")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUT / "02_accuracy_heatmap.png")
plt.close()

# ── 3. Confusion matrix: expected → detected (all models pooled) ────────
exploded = df.assign(bt=df["bias_types_detected"].str.split("|")).explode("bt")
exploded = exploded[exploded["bt"] != ""]
exploded["bt_short"] = exploded["bt"].map(BIAS_LABEL).fillna(exploded["bt"])
cm = pd.crosstab(exploded["expected_short"], exploded["bt_short"], normalize="index")
cm = cm.reindex(index=[BIAS_LABEL[b] for b in BIAS_ORDER],
                columns=[BIAS_LABEL[b] for b in BIAS_ORDER]) * 100
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues",
            cbar_kws={"label": "% of fires for that expected bias"},
            ax=ax, linewidths=0.5)
ax.set_title("Confusion — expected (rows) vs. detected (cols), pooled across all 5 models")
ax.set_xlabel("Detected bias_type")
ax.set_ylabel("Expected bias_type (human-annotated)")
plt.tight_layout()
plt.savefig(OUT / "03_confusion_pooled.png")
plt.close()

# ── 4. Per-model confusion (5-panel) ────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
for ax, mshort in zip(axes, MODEL_ORDER):
    sub = exploded[exploded["model_short"] == mshort]
    cmm = pd.crosstab(sub["expected_short"], sub["bt_short"], normalize="index")
    cmm = cmm.reindex(index=[BIAS_LABEL[b] for b in BIAS_ORDER],
                      columns=[BIAS_LABEL[b] for b in BIAS_ORDER]).fillna(0) * 100
    sns.heatmap(cmm, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
                cbar=False, ax=ax, linewidths=0.5)
    ax.set_title(mshort, fontsize=11)
    ax.set_xlabel("Detected")
    ax.set_ylabel("Expected" if ax is axes[0] else "")
fig.suptitle("Per-model confusion matrices (% within each expected row)", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "04_confusion_per_model.png", bbox_inches="tight")
plt.close()

# ── 5. Score distribution by expected_bias × model ──────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
sns.boxplot(data=df, x="expected_short", y="score", hue="model_short",
            hue_order=MODEL_ORDER, order=[BIAS_LABEL[b] for b in BIAS_ORDER],
            palette="crest", ax=ax)
ax.set_ylabel("Overall bias score (0-10)")
ax.set_xlabel("Expected bias type (human-annotated)")
ax.set_title("Score distribution per expected bias × model")
ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT / "05_score_distribution.png", bbox_inches="tight")
plt.close()

# ── 6. Consistency: per-sentence-model SD of score ──────────────────────
sd = df.groupby(["model_short", "sentence_id"])["score"].std().reset_index()
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=sd, x="model_short", y="score", order=MODEL_ORDER, palette="crest", ax=ax)
ax.set_ylabel("Per-sentence SD of score across 5 runs")
ax.set_xlabel("")
ax.set_title("Inter-run consistency — lower SD = more deterministic")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT / "06_consistency.png")
plt.close()

# ── 7. Agent firing frequency by model (which agent does each model lean on?) ──
all_agents = ["ARGUS", "LIBRA", "LENS", "QUILL", "VIGIL", "AEGIS"]
records = []
for m, g in df.groupby("model_short"):
    for ag in all_agents:
        pat = g["agents_fired"].str.contains(ag, regex=False).mean()
        records.append({"model": m, "agent": ag, "frac": pat * 100})
agf = pd.DataFrame(records)
agf_piv = agf.pivot(index="agent", columns="model", values="frac").reindex(
    index=all_agents, columns=MODEL_ORDER)
fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(agf_piv, annot=True, fmt=".0f", cmap="Purples",
            cbar_kws={"label": "% of runs where this agent fired"},
            ax=ax, linewidths=0.5)
ax.set_title("Agent firing rate per model (% of 240 runs)")
ax.set_xlabel("")
ax.set_ylabel("Agent")
plt.tight_layout()
plt.savefig(OUT / "07_agent_firing_rate.png")
plt.close()

# ── 8. "Right agent fires on right bias" diagonal ───────────────────────
# For each expected bias, does the matching agent fire?
AGENT_FOR_BIAS = {
    "confirmation_bias":      "ARGUS",
    "certainty_inflation":    "LIBRA",
    "framing_effect":         "LENS",
    "causal_inference_error": "VIGIL",
    "overgeneralisation":     "QUILL",
}
rows = []
for m, g in df.groupby("model_short"):
    for bias, agent in AGENT_FOR_BIAS.items():
        sub = g[g["expected_bias_type"] == bias]
        frac = sub["agents_fired"].str.contains(agent, regex=False).mean()
        rows.append({"model": m, "expected": BIAS_LABEL[bias], "agent": agent, "frac": frac * 100})
right_agent = pd.DataFrame(rows)
piv = right_agent.pivot(index="expected", columns="model", values="frac").reindex(
    index=[BIAS_LABEL[b] for b in BIAS_ORDER], columns=MODEL_ORDER)
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(piv, annot=True, fmt=".0f", cmap="Greens", vmin=0, vmax=100,
            cbar_kws={"label": "% of runs where the matching agent fired"},
            ax=ax, linewidths=0.5)
ax.set_title("Did the matching agent fire?  (Ideal = 100% on the right-bias row for that agent)")
ax.set_xlabel("")
ax.set_ylabel("Expected bias (→ matching agent)")
plt.tight_layout()
plt.savefig(OUT / "08_matching_agent_fire_rate.png")
plt.close()

# ── 9. Stacked: correct vs wrong-fire vs no-fire by bias_type, per model ──
def categorize(row):
    if row["n_flags"] == 0:
        return "no_fire"
    if row["correct"] == 1:
        return "correct"
    return "wrong_fire"
df["outcome"] = df.apply(categorize, axis=1)

fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
for ax, mshort in zip(axes, MODEL_ORDER):
    sub = df[df["model_short"] == mshort]
    p = pd.crosstab(sub["expected_short"], sub["outcome"], normalize="index") * 100
    p = p.reindex(index=[BIAS_LABEL[b] for b in BIAS_ORDER],
                  columns=["correct", "wrong_fire", "no_fire"]).fillna(0)
    p.plot(kind="bar", stacked=True, ax=ax,
           color=["#2a9d8f", "#e76f51", "#bdbdbd"], width=0.85, legend=(ax is axes[-1]))
    ax.set_title(mshort, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("% of runs" if ax is axes[0] else "")
    ax.set_ylim(0, 100)
    if ax is axes[-1]:
        ax.legend(bbox_to_anchor=(1.0, 1), loc="upper left", title="Outcome")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.suptitle("Outcome breakdown — correct / wrong-fire / no-fire per expected bias", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "09_outcome_breakdown.png", bbox_inches="tight")
plt.close()

# ── 10. n_flags per run distribution ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
sns.violinplot(data=df, x="model_short", y="n_flags",
               order=MODEL_ORDER, palette="crest", inner="quartile", ax=ax)
ax.set_title("Number of flags per run (lower bound = under-firing, higher = noisy)")
ax.set_ylabel("Flags per run")
ax.set_xlabel("")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT / "10_nflags_distribution.png")
plt.close()

# ── 11. AEGIS fire rate (proxy: cross-agent conflict triggered) ─────────
df["aegis_fired"] = df["agents_fired"].str.contains("AEGIS", regex=False).astype(int)
fig, ax = plt.subplots(figsize=(10, 5))
g = df.groupby(["model_short", "expected_short"])["aegis_fired"].mean().unstack() * 100
g = g.reindex(index=MODEL_ORDER, columns=[BIAS_LABEL[b] for b in BIAS_ORDER])
sns.heatmap(g, annot=True, fmt=".0f", cmap="Oranges", vmin=0, vmax=100,
            cbar_kws={"label": "% of runs where AEGIS fired"}, ax=ax, linewidths=0.5)
ax.set_title("AEGIS (conflict resolver) fire rate — how often did 2+ agents fight?")
ax.set_xlabel("Expected bias")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUT / "11_aegis_fire_rate.png")
plt.close()

# ── 12. Score vs accuracy (calibration) ─────────────────────────────────
g = df.groupby(["model_short", "expected_short"]).agg(
    mean_score=("score", "mean"), accuracy=("correct", "mean")).reset_index()
g["accuracy"] = g["accuracy"] * 100
fig, ax = plt.subplots(figsize=(8, 6))
markers = {"Confirmation":"o","Certainty":"s","Framing":"D","Causal":"^","Overgen.":"v"}
for m in MODEL_ORDER:
    sub = g[g["model_short"]==m]
    ax.scatter(sub["mean_score"], sub["accuracy"], s=120, label=m, alpha=0.8)
ax.set_xlabel("Mean reported bias score (0-10)")
ax.set_ylabel("Accuracy on that bias type (%)")
ax.set_title("Calibration — does scoring high mean detecting the right bias?")
ax.legend(loc="upper left", bbox_to_anchor=(1.01,1))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "12_score_vs_accuracy.png", bbox_inches="tight")
plt.close()

# ── 13. Outcome breakdown pooled per model (clearer single-bar view) ────
# Same three outcomes as fig 9, but bias types collapsed so each model is one
# bar. The three outcomes are mutually exclusive and exhaustive (sum = 100%),
# and the legend states exactly what each "firing" means.
order = (df.groupby("model_short")["correct"].mean()
         .reindex(MODEL_ORDER).sort_values().index.tolist())
pooled = (pd.crosstab(df["model_short"], df["outcome"], normalize="index")
          .reindex(index=order, columns=["correct", "wrong_fire", "no_fire"])
          .fillna(0) * 100)
OUTCOME_DEF = {
    "correct":    "Correct — the expected bias type was detected (success)",
    "wrong_fire": "Wrong-fire — a flag was raised, but for the wrong bias type (misattribution)",
    "no_fire":    "No-fire — nothing cleared the 0.5 confidence threshold (miss)",
}
OUTCOME_COLOR = {"correct": "#2a9d8f", "wrong_fire": "#e76f51", "no_fire": "#bdbdbd"}
fig, ax = plt.subplots(figsize=(11, 5))
left = np.zeros(len(pooled))
for col in ["correct", "wrong_fire", "no_fire"]:
    vals = pooled[col].values
    ax.barh(pooled.index, vals, left=left, height=0.7,
            color=OUTCOME_COLOR[col], label=OUTCOME_DEF[col])
    for y, (v, l) in enumerate(zip(vals, left)):
        if v >= 4:
            ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
    left += vals
ax.set_xlim(0, 100)
ax.set_xlabel("% of all 240 runs per model  (48 sentences × 5 runs)")
ax.set_ylabel("")
ax.set_title("Outcome breakdown per model — every run is exactly one of three outcomes")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False,
          fontsize=10, title="Outcome (mutually exclusive, sum to 100%)")
plt.tight_layout()
plt.savefig(OUT / "13_outcome_breakdown_pooled.png", bbox_inches="tight")
plt.close()

print("Wrote 13 figures to", OUT)
for f in sorted(OUT.glob("*.png")):
    print(" ", f.name)
