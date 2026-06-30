"""BiasScan synthesis leaderboard — detector-framed publication figures.

BiasScan is a DETECTOR of cognitive bias. Each document gets a flag density
(bias flags per 1,000 words) — length-normalised so documents of different
length are comparable.

Real data only (eval/output/leaderboard.json). 4 of 12 models produced
near-zero output due to JSON parse failures and are excluded; the 8 functioning
detectors are pooled. The replication unit is the MODEL (n=8): each model's
density is the mean over its 5 runs, and bars aggregate across models with SEM
error bars — this avoids treating 5 runs of one model as 5 independent samples.

Real per-study finding (robust across score & density, all-12 & working-8):
  MS-GUT :  GPT-Made > OG > GPT-Edited
  Nu     :  OG > GPT-Made > GPT-Edited
  BOTH   :  GPT-Edited has the LEAST detectable bias (~0)  <- the universal result
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
ROOT      = Path(__file__).resolve().parent.parent
# Override with LB_DATA / LB_OUT env vars to render a different source (e.g. leaderboard-2.json)
DATA_PATH = Path(os.environ.get("LB_DATA", ROOT / "eval" / "output" / "leaderboard.json"))
OUT_DIR   = Path(os.environ.get("LB_OUT",  ROOT / "eval" / "output" / "leaderboard_figs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = json.load(DATA_PATH.open())

DOC_WORDS = {
    "MS_GUT_OG": 726, "MS_GUT_GPT_Made": 602, "MS_GUT_GPT_Edit": 691,
    "Nu_OG": 1346, "Nu_GPT_OG": 659, "Nu_GPT_Edit": 1027,
}

STUDIES = {
    "MS-Gut Review":         ["MS_GUT_OG", "MS_GUT_GPT_Made", "MS_GUT_GPT_Edit"],
    "Nutraceuticals Review": ["Nu_OG",     "Nu_GPT_OG",       "Nu_GPT_Edit"],
}
VARIANTS = ["Human Written\n(Original)", "LLM Written", "LLM Edited"]

SHORT = {
    "meta-llama/llama-4-maverick": "Llama-4-Maverick",
    "openai/gpt-4.1-mini":         "GPT-4.1-mini",
    "google/gemini-2.5-flash":     "Gemini-2.5-Flash",
    "openai/gpt-4.1":              "GPT-4.1",
    "anthropic/claude-haiku-4.5":  "Claude-Haiku-4.5",
    "anthropic/claude-sonnet-4-5": "Claude-Sonnet-4.5",
    "deepseek/deepseek-r1":        "DeepSeek-R1",
    "google/gemini-2.5-pro":       "Gemini-2.5-Pro",
}

# Origin colours: human=steel blue, GPT-gen=warm orange (more bias), edited=green (scrubbed)
COLORS = ["#3B6BA5", "#E1812C", "#4F9D69"]

def lbl(m):
    return SHORT.get(m, m.split("/")[-1])

plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.30,
    "figure.dpi":        120,
    "savefig.dpi":       200,
})

# ── Working detectors: >=10 flags total (exclude parse-failure models) ───────
WORKING = [
    r["model"] for r in data["results"]
    if sum(run.get("n_flags", 0)
           for p in r["papers"].values() for run in p.get("runs", []))
    >= 10
]


def model_densities(paper: str):
    """Per-working-model mean flag density (flags / 1,000 words) for one paper."""
    w = DOC_WORDS[paper]
    out = {}
    for r in data["results"]:
        if r["model"] not in WORKING:
            continue
        vals = [run.get("n_flags", 0) * 1000 / w
                for run in r["papers"].get(paper, {}).get("runs", [])
                if run.get("ok")]
        if vals:
            out[r["model"]] = float(np.mean(vals))
    return out


def model_scores(paper: str):
    """Per-working-model mean bias score (0-10) for one paper — the pipeline's
    coverage-weighted output (how much text is affected x impact)."""
    out = {}
    for r in data["results"]:
        if r["model"] not in WORKING:
            continue
        vals = [(run.get("score") or 0)
                for run in r["papers"].get(paper, {}).get("runs", [])
                if run.get("ok")]
        if vals:
            out[r["model"]] = float(np.mean(vals))
    return out


def aggregate(paper: str, metric: str = "density"):
    """Mean and SEM of per-model values (model = replication unit, n<=8)."""
    src = model_scores(paper) if metric == "score" else model_densities(paper)
    vals = np.array(list(src.values()))
    if len(vals) == 0:
        return 0.0, 0.0
    sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return float(vals.mean()), float(sem)


YLAB = {"score":   "Bias score (0–10)\nimpact × coverage",
        "density": "Flag density\n(flags / 1,000 words)"}

# Global y-limits so every score plot shares one scale and every density plot
# shares another (computed from per-model maxima across all 6 documents).
SCORE_MAX = max((max(model_scores(p).values(), default=0)
                 for papers in STUDIES.values() for p in papers), default=0) or 1
DENS_MAX  = max((max(model_densities(p).values(), default=0)
                 for papers in STUDIES.values() for p in papers), default=0) or 1
SCORE_YLIM = SCORE_MAX * 1.15
DENS_YLIM  = DENS_MAX * 1.15

# ── Figure 1 — complete picture: bias SCORE (top) + flag DENSITY (bottom) ────
# Score = the pipeline's real output = how much text is affected, weighted by
# severity/confidence. Density = raw count per length. Showing both is the
# complete picture: e.g. MS-GUT OG and GPT-Gen flag a similar amount of text,
# but GPT-Gen scores ~2x higher because its bias is more severe.
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for col, (study, papers) in enumerate(STUDIES.items()):
    for row, metric in enumerate(["score", "density"]):
        ax = axes[row][col]
        means = [aggregate(p, metric)[0] for p in papers]
        sems  = [aggregate(p, metric)[1] for p in papers]
        bars = ax.bar(VARIANTS, means, color=COLORS, alpha=0.9,
                      yerr=sems, capsize=6, error_kw={"linewidth": 1.3}, zorder=3)
        ylim = SCORE_YLIM if metric == "score" else DENS_YLIM
        for b, m, s in zip(bars, means, sems):
            ax.text(b.get_x() + b.get_width()/2, m + s + ylim*0.02, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, ylim)
        ax.set_ylabel(YLAB[metric], fontsize=10)
        if row == 0:
            ax.set_title(f"{study}", fontsize=13, fontweight="bold", pad=8)
# figure heading removed — see caption list
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_complete_picture.png", bbox_inches="tight")
plt.close(fig)
print("saved fig1_complete_picture.png")


# ── Figure 2 — per-model SCORE heatmap (sentence-eval style) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4),
                         gridspec_kw={"width_ratios": [1, 1.06]})
score_max = max(max(model_scores(p).values(), default=0)
                for papers in STUDIES.values() for p in papers) or 1
for ax, (study, papers) in zip(axes, STUDIES.items()):
    rows = sorted(WORKING, key=lambda m: lbl(m))
    mat = np.array([[model_scores(p).get(m, 0.0) for p in papers] for m in rows])
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=score_max, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(VARIANTS, fontsize=9.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lbl(m) for m in rows] if ax is axes[0] else [], fontsize=9.5)
    for i in range(len(rows)):
        for j in range(3):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > score_max * 0.55 else "#333")
    ax.set_title(f"{study}", fontsize=12.5, fontweight="bold", pad=8)
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("Bias score (0–10)", fontsize=10)
# figure heading removed — see caption list
fig.savefig(OUT_DIR / "fig2_score_heatmap.png", bbox_inches="tight")
plt.close(fig)
print("saved fig2_score_heatmap.png")


# ── Figure 3 — de-biasing effect on SCORE (each model's drop to GPT-Edited) ──
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
for ax, (study, papers) in zip(axes, STUDIES.items()):
    rows = sorted(WORKING, key=lambda m: lbl(m))
    x = np.arange(3)
    for m in rows:
        y = [model_scores(p).get(m, 0.0) for p in papers]
        ax.plot(x, y, "-o", lw=1.3, ms=4.5, alpha=0.55, color="#888")
    mean_y = [aggregate(p, "score")[0] for p in papers]
    ax.plot(x, mean_y, "-o", lw=3, ms=9, color="#C0392B", label="pooled mean", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(["Human Written\n(Original)", "LLM Written", "LLM Edited"],
                                         fontsize=10)
    ax.set_ylim(0, SCORE_YLIM)
    ax.set_title(f"{study}", fontsize=12.5, fontweight="bold", pad=8)
    if ax is axes[0]:
        ax.set_ylabel("Bias score (0–10)\nimpact × coverage", fontsize=10.5)
    ax.legend(fontsize=9.5, framealpha=0.9)
# figure heading removed — see caption list
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_debiasing_effect.png", bbox_inches="tight")
plt.close(fig)
print("saved fig3_debiasing_effect.png")

print(f"\n3 headline figures -> {OUT_DIR}  (working detectors: {len(WORKING)} / {len(data['results'])})")
