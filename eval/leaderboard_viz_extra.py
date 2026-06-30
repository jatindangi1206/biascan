"""BiasScan synthesis leaderboard — extended exploratory figure set.

Everything beyond the 3 headline figures: radar fingerprints, bias-type
composition, agent firing, severity/confidence distributions, score-vs-density,
run consistency, and the ranked model leaderboard.

Real data only (eval/output/leaderboard.json), 8 working detector models
(>=10 flags; 4 parse-failure models excluded). Detector framing throughout:
a "flag" = the pipeline asserting a span carries a cognitive bias.

Override source/dest with LB_DATA / LB_OUT env vars.
"""
from __future__ import annotations
import json, os, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT      = Path(__file__).resolve().parent.parent
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
VAR_LABEL = ["Human Written (Original)", "LLM Written", "LLM Edited"]
VAR_SHORT = ["Human", "LLM Written", "LLM Edited"]
VAR_COLOR = ["#3B6BA5", "#E1812C", "#4F9D69"]

BIAS_TYPES = ["confirmation_bias", "certainty_inflation", "framing_effect",
              "overgeneralisation", "causal_inference_error"]
BIAS_LABEL = {
    "confirmation_bias": "Confirmation", "certainty_inflation": "Certainty",
    "framing_effect": "Framing", "overgeneralisation": "Overgen.",
    "causal_inference_error": "Causal",
}
AGENTS = ["ARGUS", "LIBRA", "QUILL", "LENS", "VIGIL", "AEGIS"]

SHORT = {
    "meta-llama/llama-4-maverick": "Llama-4-Maverick", "openai/gpt-4.1-mini": "GPT-4.1-mini",
    "google/gemini-2.5-flash": "Gemini-2.5-Flash", "openai/gpt-4.1": "GPT-4.1",
    "anthropic/claude-haiku-4.5": "Claude-Haiku-4.5", "anthropic/claude-sonnet-4-5": "Claude-Sonnet-4.5",
    "deepseek/deepseek-r1": "DeepSeek-R1", "google/gemini-2.5-pro": "Gemini-2.5-Pro",
}
def lbl(m): return SHORT.get(m, m.split("/")[-1])

plt.rcParams.update({
    "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.30,
    "figure.dpi": 120, "savefig.dpi": 200,
})

WORKING = [r["model"] for r in data["results"]
           if sum(run.get("n_flags", 0) for p in r["papers"].values()
                  for run in p.get("runs", [])) >= 10]
RES = {r["model"]: r for r in data["results"] if r["model"] in WORKING}


def ok_runs(model, paper):
    return [run for run in RES[model]["papers"].get(paper, {}).get("runs", []) if run.get("ok")]

def density_bt(paper, bt):
    """Mean over models of (mean over runs of flags-of-type per 1,000 words)."""
    w = DOC_WORDS[paper]; per_model = []
    for m in WORKING:
        runs = ok_runs(m, paper)
        if not runs: continue
        per_model.append(np.mean([sum(1 for f in run.get("flags", []) if f["bias_type"] == bt) * 1000 / w
                                  for run in runs]))
    return float(np.mean(per_model)) if per_model else 0.0

def density(paper):
    w = DOC_WORDS[paper]; per_model = []
    for m in WORKING:
        runs = ok_runs(m, paper)
        if not runs: continue
        per_model.append(np.mean([run.get("n_flags", 0) * 1000 / w for run in runs]))
    return per_model  # list of per-model means

# Flat flag table
FLAGS = []
for m in WORKING:
    for study, papers in STUDIES.items():
        for vi, p in enumerate(papers):
            for run in ok_runs(m, p):
                for f in run.get("flags", []):
                    FLAGS.append({"model": m, "study": study, "variant": vi,
                                  "bias_type": f["bias_type"], "agent": f["agent"],
                                  "severity": f.get("severity", "?"),
                                  "confidence": f.get("confidence", None)})


# ── fig4: radar — bias-type fingerprint by variant (per study) ───────────────
def radar_axes(ax, n):
    ang = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    return ang + ang[:1]

fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), subplot_kw=dict(polar=True))
ang = radar_axes(None, len(BIAS_TYPES))
RMAX = max((density_bt(p, bt) for papers in STUDIES.values() for p in papers
           for bt in BIAS_TYPES), default=0) or 1
for ax, (study, papers) in zip(axes, STUDIES.items()):
    for vi, p in enumerate(papers):
        vals = [density_bt(p, bt) for bt in BIAS_TYPES]
        vals += vals[:1]
        ax.plot(ang, vals, "-o", lw=2, ms=5, color=VAR_COLOR[vi], label=VAR_LABEL[vi])
        ax.fill(ang, vals, color=VAR_COLOR[vi], alpha=0.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([BIAS_LABEL[b] for b in BIAS_TYPES], fontsize=10)
    ax.set_ylim(0, RMAX)
    ax.set_title(f"{study}", fontsize=13, fontweight="bold", pad=22)
    ax.tick_params(axis="y", labelsize=8)
axes[-1].legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=10, title="Document origin")
# figure heading removed — see caption list
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_radar_biastype_by_variant.png", bbox_inches="tight")
plt.close(fig); print("saved fig4_radar_biastype_by_variant.png")


# ── fig5: per-model radar small multiples (bias-type profile, all papers) ─────
def model_bt_profile(m):
    tot = collections.Counter()
    for study, papers in STUDIES.items():
        for p in papers:
            for run in ok_runs(m, p):
                for f in run.get("flags", []):
                    tot[f["bias_type"]] += 1
    return [tot[b] for b in BIAS_TYPES]

rows = sorted(WORKING, key=lbl)
fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), subplot_kw=dict(polar=True))
vmax = max(max(model_bt_profile(m)) for m in rows) or 1
for ax, m in zip(axes.flat, rows):
    vals = model_bt_profile(m); vals += vals[:1]
    ax.plot(ang, vals, "-o", lw=1.8, ms=4, color="#C0392B")
    ax.fill(ang, vals, color="#C0392B", alpha=0.18)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([BIAS_LABEL[b] for b in BIAS_TYPES], fontsize=8)
    ax.set_ylim(0, vmax); ax.set_yticklabels([])
    ax.set_title(lbl(m), fontsize=11, fontweight="bold", pad=14)
# figure heading removed — see caption list
fig.tight_layout()
fig.savefig(OUT_DIR / "fig5_radar_per_model.png", bbox_inches="tight")
plt.close(fig); print("saved fig5_radar_per_model.png")


# ── fig6: bias-type composition stacked bars per variant (normalised %) ──────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
bt_colors = plt.cm.Set2(np.linspace(0, 1, len(BIAS_TYPES)))
for ax, (study, papers) in zip(axes, STUDIES.items()):
    bottoms = np.zeros(3)
    comp = np.array([[density_bt(p, bt) for p in papers] for bt in BIAS_TYPES])  # bt x variant
    totals = comp.sum(axis=0); totals[totals == 0] = 1
    pct = comp / totals * 100
    for bi, bt in enumerate(BIAS_TYPES):
        ax.bar(VAR_LABEL, pct[bi], bottom=bottoms, color=bt_colors[bi],
               label=BIAS_LABEL[bt], edgecolor="white", linewidth=0.6)
        bottoms += pct[bi]
    ax.set_title(f"{study}", fontsize=12.5, fontweight="bold")
    ax.set_ylabel("% of detected bias" if ax is axes[0] else "")
    ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), rotation=12)
axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9.5, title="Bias type")
# figure heading removed — see caption list
fig.tight_layout()
fig.savefig(OUT_DIR / "fig6_biastype_composition.png", bbox_inches="tight")
plt.close(fig); print("saved fig6_biastype_composition.png")


# ── fig7: agent firing heatmap (agent × variant) per study ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.06]})
for ax, (study, papers) in zip(axes, STUDIES.items()):
    mat = np.zeros((len(AGENTS), 3))
    for f in FLAGS:
        if f["study"] != study: continue
        if f["agent"] in AGENTS:
            mat[AGENTS.index(f["agent"]), f["variant"]] += 1
    im = ax.imshow(mat, cmap="Purples", aspect="auto", vmin=0, vmax=max(mat.max(), 1))
    ax.set_xticks(range(3)); ax.set_xticklabels(VAR_LABEL, fontsize=9.5, rotation=12)
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels(AGENTS if ax is axes[0] else [], fontsize=10)
    for i in range(len(AGENTS)):
        for j in range(3):
            ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center", fontsize=10,
                    color="white" if mat[i, j] > mat.max()*0.55 else "#333")
    ax.set_title(f"{study}", fontsize=12.5, fontweight="bold")
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cbar.set_label("Flags raised", fontsize=10)
# figure heading removed — see caption list
fig.savefig(OUT_DIR / "fig7_agent_firing.png", bbox_inches="tight")
plt.close(fig); print("saved fig7_agent_firing.png")


# ── fig8: severity composition per variant (stacked) ─────────────────────────
SEV = ["low", "medium", "high"]; SEV_COLOR = ["#A8D5BA", "#F2C14E", "#E07A5F"]
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, (study, papers) in zip(axes, STUDIES.items()):
    counts = np.zeros((len(SEV), 3))
    for f in FLAGS:
        if f["study"] != study: continue
        if f["severity"] in SEV:
            counts[SEV.index(f["severity"]), f["variant"]] += 1
    tot = counts.sum(axis=0); tot[tot == 0] = 1
    pct = counts / tot * 100; bottoms = np.zeros(3)
    for si, s in enumerate(SEV):
        ax.bar(VAR_LABEL, pct[si], bottom=bottoms, color=SEV_COLOR[si], label=s.capitalize(),
               edgecolor="white")
        bottoms += pct[si]
    ax.set_title(f"{study}", fontsize=12.5, fontweight="bold")
    ax.set_ylabel("% of flags" if ax is axes[0] else ""); ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), rotation=12)
axes[-1].legend(title="Severity", fontsize=9.5, loc="upper right")
# figure heading removed — see caption list
fig.tight_layout()
fig.savefig(OUT_DIR / "fig8_severity_mix.png", bbox_inches="tight")
plt.close(fig); print("saved fig8_severity_mix.png")


# ── fig9: confidence distribution per variant (box) ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
groups, positions, colors, ticklabels = [], [], [], []
pos = 0
for study, papers in STUDIES.items():
    for vi in range(3):
        conf = [f["confidence"] for f in FLAGS
                if f["study"] == study and f["variant"] == vi and f["confidence"] is not None]
        groups.append(conf if conf else [np.nan]); positions.append(pos)
        colors.append(VAR_COLOR[vi]); ticklabels.append(f"{study}\n{VAR_SHORT[vi]}")
        pos += 1
    pos += 0.6
bp = ax.boxplot(groups, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_xticks(positions); ax.set_xticklabels(ticklabels, fontsize=8.5)
ax.set_ylabel("Flag confidence"); ax.grid(True, axis="y", alpha=0.3)
# figure heading removed — see caption list
fig.tight_layout()
fig.savefig(OUT_DIR / "fig9_confidence.png", bbox_inches="tight")
plt.close(fig); print("saved fig9_confidence.png")


# ── fig10: score vs density scatter (per model × paper) ──────────────────────
fig, ax = plt.subplots(figsize=(8.5, 6))
markers = dict(zip(STUDIES, ["o", "s"]))
for study, papers in STUDIES.items():
    for vi, p in enumerate(papers):
        for m in WORKING:
            runs = ok_runs(m, p)
            if not runs: continue
            sc = np.mean([(run.get("score") or 0) for run in runs])
            de = np.mean([run.get("n_flags", 0) * 1000 / DOC_WORDS[p] for run in runs])
            ax.scatter(de, sc, s=70, marker=markers[study], color=VAR_COLOR[vi],
                       alpha=0.75, edgecolor="white", linewidth=0.5)
from matplotlib.lines import Line2D
leg1 = [Line2D([0],[0], marker="o", color="w", markerfacecolor=VAR_COLOR[i], markersize=10,
               label=VAR_LABEL[i]) for i in range(3)]
leg2 = [Line2D([0],[0], marker=markers[s], color="gray", linestyle="", markersize=9, label=s)
        for s in STUDIES]
ax.legend(handles=leg1+leg2, fontsize=9, loc="upper left")
ax.set_xlabel("Flag density (flags / 1,000 words)"); ax.set_ylabel("Overall bias score (0–10)")
# figure heading removed — see caption list
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig10_score_vs_density.png", bbox_inches="tight")
plt.close(fig); print("saved fig10_score_vs_density.png")


# ── fig11: run-to-run consistency (SD of density per model) ──────────────────
fig, ax = plt.subplots(figsize=(9.5, 5))
rows = sorted(WORKING, key=lbl)
sds = []
for m in rows:
    per_run = []
    for study, papers in STUDIES.items():
        for p in papers:
            for run in ok_runs(m, p):
                per_run.append(run.get("n_flags", 0) * 1000 / DOC_WORDS[p])
    # SD across runs within each paper, averaged
    paper_sds = []
    for study, papers in STUDIES.items():
        for p in papers:
            d = [run.get("n_flags", 0) * 1000 / DOC_WORDS[p] for run in ok_runs(m, p)]
            if len(d) >= 2: paper_sds.append(np.std(d, ddof=1))
    sds.append(np.mean(paper_sds) if paper_sds else 0)
bars = ax.bar([lbl(m) for m in rows], sds, color="#6A8CAF", alpha=0.88, zorder=3)
for b, v in zip(bars, sds):
    ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Mean within-document SD of flag density\n(across 5 runs; lower = more consistent)")
# figure heading removed — see caption list
plt.setp(ax.get_xticklabels(), rotation=22, ha="right")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig11_consistency.png", bbox_inches="tight")
plt.close(fig); print("saved fig11_consistency.png")


# ── fig12: ranked model leaderboard (mean density across all 6 docs) ─────────
fig, ax = plt.subplots(figsize=(9, 6))
rank = []
for m in WORKING:
    ds = []
    for study, papers in STUDIES.items():
        for p in papers:
            runs = ok_runs(m, p)
            if runs: ds.append(np.mean([run.get("n_flags", 0) * 1000 / DOC_WORDS[p] for run in runs]))
    rank.append((lbl(m), np.mean(ds) if ds else 0))
rank.sort(key=lambda x: x[1])
names = [r[0] for r in rank]; vals = [r[1] for r in rank]
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
bars = ax.barh(names, vals, color=cmap, alpha=0.9)
for b, v in zip(bars, vals):
    ax.text(v+0.02, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Mean flag density across all 6 documents (flags / 1,000 words)")
# figure heading removed — see caption list
ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig12_model_leaderboard.png", bbox_inches="tight")
plt.close(fig); print("saved fig12_model_leaderboard.png")

print(f"\n9 extra figures -> {OUT_DIR}  (working detectors: {len(WORKING)})")
