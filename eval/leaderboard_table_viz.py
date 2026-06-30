"""Leaderboard figures from leaderboard-2.json — ranked model performance."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "eval" / "output" / "leaderboard-2.json"
OUT_DIR   = ROOT / "eval" / "output" / "leaderboard_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

data   = json.load(DATA_PATH.open())
PAPERS = data["papers"]

EXCLUDE = {"openai/gpt-5", "openai/gpt-5.5"}

SHORT = {
    "meta-llama/llama-4-maverick":       "Llama 4 Maverick",
    "openai/gpt-4.1-mini":               "GPT-4.1 Mini",
    "google/gemini-2.5-flash":           "Gemini 2.5 Flash",
    "openai/gpt-4.1":                    "GPT-4.1",
    "anthropic/claude-haiku-4.5":        "Claude Haiku 4.5",
    "deepseek/deepseek-r1":              "DeepSeek R1",
    "google/gemini-2.5-pro":             "Gemini 2.5 Pro",
    "anthropic/claude-sonnet-4-5":       "Claude Sonnet 4.5",
    "x-ai/grok-4.3":                     "Grok 4.3",
    "z-ai/glm-5.2":                      "GLM 5.2",
    "google/gemini-2.5-pro-preview":     "Gemini 2.5 Pro Preview",
}

# Papers excluding the edit variants (always ~0, not informative for ranking)
MS_GUT_PAPERS = ["MS_GUT_OG", "MS_GUT_GPT_Made"]
NU_PAPERS      = ["Nu_OG",     "Nu_GPT_OG"]
ALL_ACTIVE     = MS_GUT_PAPERS + NU_PAPERS

LABELS = {
    "MS_GUT_OG":       "MS-Gut Review — Human Written",
    "MS_GUT_GPT_Made": "MS-Gut Review — LLM Written",
    "Nu_OG":           "Nutraceuticals Review — Human Written",
    "Nu_GPT_OG":       "Nutraceuticals Review — LLM Written",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Build rows ───────────────────────────────────────────────────────────────
rows = []
for r in data["results"]:
    mid = r["model"]
    if mid in EXCLUDE:
        continue
    scores = {p: r["papers"].get(p, {}).get("mean", 0) for p in ALL_ACTIVE}
    avg = np.mean(list(scores.values()))
    rows.append({
        "model":  SHORT.get(mid, mid.split("/")[-1]),
        "avg":    avg,
        "scores": scores,
    })

rows.sort(key=lambda x: -x["avg"])


# ── Figure A: ranked horizontal bar (overall avg across both studies) ────────
fig, ax = plt.subplots(figsize=(8, max(5, len(rows) * 0.55 + 1.5)))

models = [r["model"] for r in rows]
avgs   = [r["avg"]   for r in rows]

# colour by rank quartile
n = len(rows)
colors = []
for i in range(n):
    if i < n // 3:
        colors.append("#2C6FAC")      # top — dark blue
    elif i < 2 * n // 3:
        colors.append("#5BA3D9")      # mid — medium blue
    else:
        colors.append("#A8C8E8")      # bottom — light blue

bars = ax.barh(models[::-1], avgs[::-1], color=colors[::-1], alpha=0.88, height=0.6)
for bar, val in zip(bars, avgs[::-1]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=9.5, fontweight="bold")

ax.set_xlabel("Mean Bias Score (0–10)", fontsize=11)
# figure heading removed — see caption list
ax.xaxis.grid(True, alpha=0.35)
ax.set_axisbelow(True)
ax.set_xlim(0, max(avgs) * 1.25 if avgs else 1)
fig.tight_layout()
fig.savefig(OUT_DIR / "5_leaderboard_overall.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 5_leaderboard_overall.png")


# ── Figure B: grouped score table — all 4 active papers per model ─────────
fig, ax = plt.subplots(figsize=(11, max(5, len(rows) * 0.6 + 2)))

paper_colors = ["#4C72B0", "#DD8452", "#C44E52", "#8172B3"]
y      = np.arange(len(rows))
height = 0.18
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, (paper, color) in enumerate(zip(ALL_ACTIVE, paper_colors)):
    vals = [r["scores"][paper] for r in rows]
    ax.barh(y + offsets[i] * height, vals, height,
            label=LABELS[paper], color=color, alpha=0.82)

ax.set_yticks(y)
ax.set_yticklabels([r["model"] for r in rows], fontsize=9.5)
ax.set_xlabel("Bias Score (0–10)", fontsize=11)
# figure heading removed — see caption list
ax.legend(title="Document", fontsize=9.5, loc="lower right", framealpha=0.9)
ax.xaxis.grid(True, alpha=0.35)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "5_leaderboard_breakdown.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 5_leaderboard_breakdown.png")


# ── Figure C: study-split leaderboard (MS_GUT avg | Nu avg) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(rows) * 0.55 + 1.5)), sharey=True)

study_avgs = {
    "MS-Gut Review":         [np.mean([r["scores"][p] for p in MS_GUT_PAPERS]) for r in rows],
    "Nutraceuticals Review": [np.mean([r["scores"][p] for p in NU_PAPERS]) for r in rows],
}
xmax = max((v for vals in study_avgs.values() for v in vals), default=1) * 1.3

for ax, (study_label, color) in zip(
    axes, [("MS-Gut Review", "#4C72B0"), ("Nutraceuticals Review", "#C44E52")],
):
    avgs_study = study_avgs[study_label]
    ax.barh([r["model"] for r in rows][::-1], avgs_study[::-1],
            color=color, alpha=0.82, height=0.6)
    for i, val in enumerate(avgs_study[::-1]):
        ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Mean Bias Score", fontsize=11)
    ax.set_title(study_label, fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlim(0, xmax)

# figure heading removed — see caption list
fig.tight_layout()
fig.savefig(OUT_DIR / "5_leaderboard_per_study.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 5_leaderboard_per_study.png")

print(f"\nDone — leaderboard figures in {OUT_DIR}")
