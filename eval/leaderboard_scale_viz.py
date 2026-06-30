"""Scale & score-component figures for the document-level benchmark.

Shows the pieces behind the bias score — document word count, span coverage,
and severity-weighted impact — per study (MS-GUT, Nu) and per document origin
(Human Written / LLM Written / LLM Edited), as normal bar plots and a radar.

Coverage and impact are recomputed from the stored flags using the current
SCORING.md definition (the stored `score` field predates that formula):
  impact   = Σ confidence × severity_weight  +  0.15 × (unique_bias_types − 1)
             severity weight: high=3, medium=2, low=1
  coverage = union of flagged words / total words   (flagged spans located by
             substring-matching each flag's flagged_text in the source .txt)
  score    = min(10, impact × coverage × 3.33)      (0–10)

Replication unit = MODEL (mean over its 5 ok runs); bars aggregate across the
8 working detectors with SEM error bars.

Run:  python -m eval.leaderboard_scale_viz
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval._doc_metrics import (
    STUDIES, VARIANTS, COLORS, DOC_WORDS, working_models, ok_runs,
    run_coverage, run_impact, run_density, run_score, run_conf,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = json.load((ROOT / "eval" / "output" / "leaderboard.json").open())
OUT  = ROOT / "eval" / "output" / "leaderboard_figs"
OUT.mkdir(parents=True, exist_ok=True)

RES = {r["model"]: r for r in DATA["results"]}
WORKING = working_models(DATA)

plt.rcParams.update({
    "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.30,
    "figure.dpi": 120, "savefig.dpi": 200,
})


def cell(paper, fn):
    """Mean and SEM across working models of each model's mean over its ok runs."""
    per_model = []
    for m in WORKING:
        runs = ok_runs(RES, m, paper)
        if runs:
            per_model.append(np.mean([fn(r, paper) if fn.__code__.co_argcount == 2 else fn(r)
                                      for r in runs]))
    v = np.array(per_model, float)
    if len(v) == 0:
        return 0.0, 0.0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


def grouped_bar(ax, values, sems=None, ylabel="", ylim=None):
    """values/sems: dict study -> [v_human, v_written, v_edited]."""
    studies = list(values)
    x = np.arange(len(studies)); w = 0.26
    for vi in range(3):
        vals = [values[s][vi] for s in studies]
        err  = [sems[s][vi] for s in studies] if sems else None
        bars = ax.bar(x + (vi - 1) * w, vals, w, color=COLORS[vi], label=VARIANTS[vi],
                      yerr=err, capsize=4, error_kw={"linewidth": 1.1}, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (ylim or max(vals))*0.015,
                    f"{v:.2f}" if v < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(studies, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10.5)
    if ylim:
        ax.set_ylim(0, ylim)


# ── Fig 13 — document word count (exact scale) ───────────────────────────────
wc = {s: [DOC_WORDS[p] for p in papers] for s, papers in STUDIES.items()}
fig, ax = plt.subplots(figsize=(8, 5))
grouped_bar(ax, wc, ylabel="Document length (words)",
            ylim=max(v for vs in wc.values() for v in vs) * 1.18)
ax.legend(fontsize=9.5, title="Document origin")
# figure heading removed — see caption list
fig.tight_layout(); fig.savefig(OUT / "fig13_wordcount.png", bbox_inches="tight"); plt.close(fig)
print("saved fig13_wordcount.png")


# ── Fig 14 — score components: coverage | impact | score ─────────────────────
metrics = [
    ("coverage", run_coverage, "Span coverage (fraction of words flagged)"),
    ("impact",   run_impact,   "Severity-weighted impact (Σ conf × severity)"),
    ("score",    run_score,    "Bias score (0–10) = impact × coverage × 3.33"),
]
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, (name, fn, ylab) in zip(axes, metrics):
    vals = {s: [cell(p, fn)[0] for p in papers] for s, papers in STUDIES.items()}
    sems = {s: [cell(p, fn)[1] for p in papers] for s, papers in STUDIES.items()}
    ymax = max(v + e for s in vals for v, e in zip(vals[s], sems[s])) * 1.2 or 1
    grouped_bar(ax, vals, sems, ylabel=ylab, ylim=ymax)
    # panel heading removed — see caption list (panels are coverage | impact | score)
axes[-1].legend(fontsize=9, title="Document origin")
fig.tight_layout(); fig.savefig(OUT / "fig14_score_components.png", bbox_inches="tight"); plt.close(fig)
print("saved fig14_score_components.png")


# ── Fig 15 — radar: scale & detection fingerprint per origin (per study) ──────
AXES = [("Coverage", run_coverage), ("Impact", run_impact), ("Flag density", run_density),
        ("Score", run_score), ("Confidence", run_conf)]
# global per-axis max for normalisation (so each axis is 0..1, comparable)
axis_max = []
for _, fn in AXES:
    mx = max(cell(p, fn)[0] for papers in STUDIES.values() for p in papers)
    axis_max.append(mx or 1.0)
ang = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False).tolist(); ang += ang[:1]
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), subplot_kw=dict(polar=True))
for ax, (study, papers) in zip(axes, STUDIES.items()):
    for vi, p in enumerate(papers):
        vals = [cell(p, fn)[0] / axis_max[i] for i, (_, fn) in enumerate(AXES)]
        vals += vals[:1]
        ax.plot(ang, vals, "-o", lw=2, ms=5, color=COLORS[vi], label=VARIANTS[vi])
        ax.fill(ang, vals, color=COLORS[vi], alpha=0.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([a for a, _ in AXES], fontsize=10)
    ax.set_ylim(0, 1.0); ax.set_yticklabels([])
    ax.set_title(study, fontsize=13, fontweight="bold", pad=22)
axes[-1].legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9.5, title="Document origin")
# figure heading removed — see caption list
fig.tight_layout(); fig.savefig(OUT / "fig15_components_radar.png", bbox_inches="tight"); plt.close(fig)
print("saved fig15_components_radar.png")

print(f"\n3 scale figures -> {OUT}  (working detectors: {len(WORKING)})")
