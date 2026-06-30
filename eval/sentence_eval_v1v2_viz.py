"""V1 vs V2 prompt-version comparison plots.

Reads:
  eval/output/sentence_eval_per_run.csv      (v1: 5 runs per cell)
  eval/output/sentence_eval_v2_per_run.csv   (v2: 1 run per cell)

Writes PNGs to eval/output/sentence_eval_v1v2_figs/.

Comparison convention: we compare v2 (1 run) against v1 averaged across its 5
runs. This is statistically fairer than comparing v2 to a single v1 run.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
V1_CSV = ROOT / "eval" / "output" / "sentence_eval_per_run.csv"
V2_CSV = ROOT / "eval" / "output" / "sentence_eval_v2_per_run.csv"
OUT = ROOT / "eval" / "output" / "sentence_eval_v1v2_figs"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150

SHORT = {
    "openai/gpt-4.1-nano":               "GPT-4.1",
    "anthropic/claude-3-haiku":          "Claude-Haiku-4.5",
    "google/gemini-2.5-flash-lite":      "Gemini-2.5-Flash",
    "deepseek/deepseek-v4-flash":        "DeepSeek-R1",
    "meta-llama/llama-3.3-70b-instruct": "Llama-4-Maverick",
}
MODEL_ORDER = ["GPT-4.1","Claude-Haiku-4.5","Gemini-2.5-Flash","DeepSeek-R1","Llama-4-Maverick"]
BIAS_ORDER = ["confirmation_bias","certainty_inflation","framing_effect",
              "causal_inference_error","overgeneralisation"]
BIAS_LABEL = {"confirmation_bias":"Confirmation","certainty_inflation":"Certainty",
              "framing_effect":"Framing","causal_inference_error":"Causal",
              "overgeneralisation":"Overgen."}

def prep(df):
    df = df.copy()
    df["m"] = df["model"].map(SHORT)
    df["agents_fired"] = df["agents_fired"].fillna("")
    df["bias_types_detected"] = df["bias_types_detected"].fillna("")
    df["any_fire"] = (df["n_flags"] > 0).astype(int)
    df["expected_short"] = df["expected_bias_type"].map(BIAS_LABEL)
    return df

v1 = prep(pd.read_csv(V1_CSV))
v2 = prep(pd.read_csv(V2_CSV))

# ── 1. Detection / Attribution / Score side-by-side per model ─────────────
hdr_v1 = v1.groupby("m").agg(det=("any_fire","mean"), att=("correct","mean"),
                              sc=("score","mean")).reindex(MODEL_ORDER) * [100,100,1]
hdr_v2 = v2.groupby("m").agg(det=("any_fire","mean"), att=("correct","mean"),
                              sc=("score","mean")).reindex(MODEL_ORDER) * [100,100,1]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x = np.arange(len(MODEL_ORDER))
w = 0.36

for ax, col, ylabel, title, color_v1, color_v2 in [
    (axes[0], "det", "% of runs", "Detection rate (any agent fired)", "#264653", "#2a9d8f"),
    (axes[1], "att", "% of runs", "Attribution accuracy (right bias type)", "#264653", "#2a9d8f"),
    (axes[2], "sc",  "Mean bias score (0-10)", "Mean bias score", "#264653", "#e76f51"),
]:
    b1 = ax.bar(x - w/2, hdr_v1[col], w, label="v1 prompts", color=color_v1)
    b2 = ax.bar(x + w/2, hdr_v2[col], w, label="v2 prompts", color=color_v2)
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{bar.get_height():.1f}", ha="center", fontsize=9)
    for bar in b2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{bar.get_height():.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if col != "sc":
        ax.set_ylim(0, 110)
    else:
        ax.set_ylim(0, 11)
fig.suptitle("v1 → v2 prompt comparison — same 48 sentences, same 5 models", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "01_headline_v1_v2.png", bbox_inches="tight")
plt.close()

# ── 2. Per-bias attribution heatmap: v1 vs v2 ────────────────────────────
acc_v1 = v1.pivot_table(values="correct", index="m", columns="expected_bias_type", aggfunc="mean")\
           .reindex(index=MODEL_ORDER, columns=BIAS_ORDER) * 100
acc_v2 = v2.pivot_table(values="correct", index="m", columns="expected_bias_type", aggfunc="mean")\
           .reindex(index=MODEL_ORDER, columns=BIAS_ORDER) * 100
delta = acc_v2 - acc_v1

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
sns.heatmap(acc_v1.rename(columns=BIAS_LABEL), annot=True, fmt=".0f", cmap="RdYlGn",
            vmin=0, vmax=100, ax=axes[0], linewidths=0.5,
            cbar_kws={"label":"Accuracy %"})
axes[0].set_title("v1 attribution (% correct, 5-run avg)")
axes[0].set_xlabel(""); axes[0].set_ylabel("")

sns.heatmap(acc_v2.rename(columns=BIAS_LABEL), annot=True, fmt=".0f", cmap="RdYlGn",
            vmin=0, vmax=100, ax=axes[1], linewidths=0.5,
            cbar_kws={"label":"Accuracy %"})
axes[1].set_title("v2 attribution (% correct, 1 run)")
axes[1].set_xlabel(""); axes[1].set_ylabel("")

sns.heatmap(delta.rename(columns=BIAS_LABEL), annot=True, fmt="+.0f", cmap="RdBu",
            center=0, vmin=-80, vmax=80, ax=axes[2], linewidths=0.5,
            cbar_kws={"label":"Δ pp (v2 − v1)"})
axes[2].set_title("Δ attribution (v2 − v1, percentage points)")
axes[2].set_xlabel(""); axes[2].set_ylabel("")

plt.tight_layout()
plt.savefig(OUT / "02_attribution_heatmap_v1_v2.png", bbox_inches="tight")
plt.close()

# ── 3. Pooled confusion: v1 vs v2 ────────────────────────────────────────
def conf(df):
    exp = df.assign(bt=df["bias_types_detected"].str.split("|")).explode("bt")
    exp = exp[exp["bt"]!=""]
    cm = pd.crosstab(exp["expected_bias_type"], exp["bt"], normalize="index")
    return cm.reindex(index=BIAS_ORDER, columns=BIAS_ORDER).fillna(0) * 100

c1 = conf(v1).rename(index=BIAS_LABEL, columns=BIAS_LABEL)
c2 = conf(v2).rename(index=BIAS_LABEL, columns=BIAS_LABEL)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(c1, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
            ax=axes[0], linewidths=0.5, cbar_kws={"label":"% within expected row"})
axes[0].set_title("v1 confusion (pooled across 5 models)")
axes[0].set_xlabel("Detected"); axes[0].set_ylabel("Expected")
sns.heatmap(c2, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
            ax=axes[1], linewidths=0.5, cbar_kws={"label":"% within expected row"})
axes[1].set_title("v2 confusion (pooled across 5 models)")
axes[1].set_xlabel("Detected"); axes[1].set_ylabel("")
plt.tight_layout()
plt.savefig(OUT / "03_confusion_v1_v2.png", bbox_inches="tight")
plt.close()

# ── 4. Agent on-target % v1 vs v2 ────────────────────────────────────────
AGENT_TARGETS = {"ARGUS":"confirmation_bias","LIBRA":"certainty_inflation",
                 "QUILL":"framing_effect","VIGIL":"causal_inference_error",
                 "LENS":"overgeneralisation"}
rows = []
for ver, df in [("v1", v1), ("v2", v2)]:
    for ag, tgt in AGENT_TARGETS.items():
        fired = df[df["agents_fired"].str.contains(ag, regex=False)]
        n = len(fired)
        on = (fired["expected_bias_type"]==tgt).mean()*100 if n else 0
        rows.append({"version":ver,"agent":ag,"target":BIAS_LABEL[tgt],"fires":n,"on_target":on})
agf = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(data=agf, x="agent", y="on_target", hue="version",
            palette={"v1":"#264653","v2":"#2a9d8f"}, ax=axes[0])
axes[0].set_ylabel("On-target % of fires")
axes[0].set_xlabel("")
axes[0].set_title("Agent precision: when this agent fired, was the expected bias correct?")
axes[0].set_ylim(0, 100)
for c in axes[0].containers:
    axes[0].bar_label(c, fmt="%.0f", padding=2, fontsize=9)

sns.barplot(data=agf, x="agent", y="fires", hue="version",
            palette={"v1":"#264653","v2":"#2a9d8f"}, ax=axes[1])
axes[1].set_ylabel("Total fires")
axes[1].set_xlabel("")
axes[1].set_title("Agent activity: how often did this agent fire across all runs?")
for c in axes[1].containers:
    axes[1].bar_label(c, fmt="%.0f", padding=2, fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "04_agent_precision_activity_v1_v2.png", bbox_inches="tight")
plt.close()

# ── 5. n_flags + AEGIS rate per model: v1 vs v2 ──────────────────────────
v1["aegis"] = v1["agents_fired"].str.contains("AEGIS", regex=False).astype(int)
v2["aegis"] = v2["agents_fired"].str.contains("AEGIS", regex=False).astype(int)
nf_v1 = v1.groupby("m")["n_flags"].mean().reindex(MODEL_ORDER)
nf_v2 = v2.groupby("m")["n_flags"].mean().reindex(MODEL_ORDER)
ae_v1 = v1.groupby("m")["aegis"].mean().reindex(MODEL_ORDER) * 100
ae_v2 = v2.groupby("m")["aegis"].mean().reindex(MODEL_ORDER) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(MODEL_ORDER))
w = 0.36
axes[0].bar(x-w/2, nf_v1, w, label="v1", color="#264653")
axes[0].bar(x+w/2, nf_v2, w, label="v2", color="#2a9d8f")
axes[0].set_xticks(x); axes[0].set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
axes[0].set_ylabel("Mean flags per run")
axes[0].set_title("How loudly is the pipeline firing? (mean n_flags)")
axes[0].legend()
for i,(a,b) in enumerate(zip(nf_v1, nf_v2)):
    axes[0].text(i-w/2, a+0.03, f"{a:.2f}", ha="center", fontsize=9)
    axes[0].text(i+w/2, b+0.03, f"{b:.2f}", ha="center", fontsize=9, fontweight="bold")

axes[1].bar(x-w/2, ae_v1, w, label="v1", color="#264653")
axes[1].bar(x+w/2, ae_v2, w, label="v2", color="#e76f51")
axes[1].set_xticks(x); axes[1].set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
axes[1].set_ylabel("% of runs where AEGIS fired")
axes[1].set_title("AEGIS conflict-resolver activation rate")
axes[1].legend()
axes[1].set_ylim(0, 110)
for i,(a,b) in enumerate(zip(ae_v1, ae_v2)):
    axes[1].text(i-w/2, a+1, f"{a:.0f}", ha="center", fontsize=9)
    axes[1].text(i+w/2, b+1, f"{b:.0f}", ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "05_nflags_aegis_v1_v2.png", bbox_inches="tight")
plt.close()

# ── 6. Per-sentence heatmap, side-by-side ────────────────────────────────
def per_sent(df):
    """For each sentence × model, fraction of runs where correct."""
    return df.groupby(["sentence_id","expected_bias_type","m"])["correct"].mean().reset_index()

ps_v1 = per_sent(v1).pivot_table(index=["expected_bias_type","sentence_id"],
                                  columns="m", values="correct").reindex(columns=MODEL_ORDER)
ps_v2 = per_sent(v2).pivot_table(index=["expected_bias_type","sentence_id"],
                                  columns="m", values="correct").reindex(columns=MODEL_ORDER)

fig, axes = plt.subplots(1, 2, figsize=(13, 14), sharey=True)
sns.heatmap(ps_v1, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=1,
            ax=axes[0], cbar=False, linewidths=0.3)
axes[0].set_title("v1: fraction of 5 runs correct per (sentence × model)")
axes[0].set_xlabel("")
sns.heatmap(ps_v2, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=1,
            ax=axes[1], cbar_kws={"label":"correct (0/1)"}, linewidths=0.3)
axes[1].set_title("v2: correct (0/1) per (sentence × model)")
axes[1].set_xlabel("")
plt.tight_layout()
plt.savefig(OUT / "06_per_sentence_v1_v2.png", bbox_inches="tight")
plt.close()

print("Wrote 6 figures to", OUT)
for f in sorted(OUT.glob("*.png")):
    print(" ", f.name)
