"""Study-wise figure generator — one folder per study, every graph single-study.

Renders the full figure set separately for each study into:
  eval/output/leaderboard_figs_MS_GUT/
  eval/output/leaderboard_figs_Nu/

Real data only (eval/output/leaderboard.json), 8 working detector models
(>=10 flags; 4 parse-failure models excluded). Detector framing throughout.
Override source with LB_DATA env var.
"""
from __future__ import annotations
import json, os, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.environ.get("LB_DATA", ROOT / "eval" / "output" / "leaderboard.json"))
OUT_BASE  = ROOT / "eval" / "output"
data = json.load(DATA_PATH.open())

DOC_WORDS = {
    "MS_GUT_OG": 726, "MS_GUT_GPT_Made": 602, "MS_GUT_GPT_Edit": 691,
    "Nu_OG": 1346, "Nu_GPT_OG": 659, "Nu_GPT_Edit": 1027,
}
STUDIES = {
    "MS_GUT": {"papers": ["MS_GUT_OG", "MS_GUT_GPT_Made", "MS_GUT_GPT_Edit"], "title": "MS-GUT"},
    "Nu":     {"papers": ["Nu_OG", "Nu_GPT_OG", "Nu_GPT_Edit"],              "title": "Nu"},
}
VARIANTS  = ["Human-Written\n(original)", "GPT-Generated", "GPT-Edited\n(de-biased)"]
VSHORT    = ["OG", "GPT-Gen", "GPT-Edit"]
VAR_COLOR = ["#3B6BA5", "#E1812C", "#4F9D69"]

BIAS_TYPES = ["confirmation_bias", "certainty_inflation", "framing_effect",
              "overgeneralisation", "causal_inference_error"]
BIAS_LABEL = {"confirmation_bias": "Confirmation", "certainty_inflation": "Certainty",
              "framing_effect": "Framing", "overgeneralisation": "Overgen.",
              "causal_inference_error": "Causal"}
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

def ok_runs(m, paper): return [r for r in RES[m]["papers"].get(paper, {}).get("runs", []) if r.get("ok")]

def model_density(m, paper):
    runs = ok_runs(m, paper)
    return float(np.mean([r.get("n_flags", 0) * 1000 / DOC_WORDS[paper] for r in runs])) if runs else None
def model_score(m, paper):
    runs = ok_runs(m, paper)
    return float(np.mean([(r.get("score") or 0) for r in runs])) if runs else None

def agg(paper, metric):
    f = model_score if metric == "score" else model_density
    vals = np.array([v for m in WORKING if (v := f(m, paper)) is not None])
    if len(vals) == 0: return 0.0, 0.0
    return float(vals.mean()), float(vals.std(ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

def density_bt(paper, bt):
    w = DOC_WORDS[paper]; per = []
    for m in WORKING:
        runs = ok_runs(m, paper)
        if runs:
            per.append(np.mean([sum(1 for fl in r.get("flags", []) if fl["bias_type"] == bt)*1000/w for r in runs]))
    return float(np.mean(per)) if per else 0.0

YLAB = {"score": "Bias score (0–10)\nimpact × coverage", "density": "Flag density\n(flags / 1,000 words)"}


def render(study_key):
    spec = STUDIES[study_key]; papers = spec["papers"]; title = spec["title"]
    OUT = OUT_BASE / f"leaderboard_figs_{study_key}"
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"): old.unlink()
    rows = sorted(WORKING, key=lbl)
    ang = np.linspace(0, 2*np.pi, len(BIAS_TYPES), endpoint=False).tolist(); ang += ang[:1]

    # flat flags for this study
    FL = [{"variant": vi, "bias_type": fl["bias_type"], "agent": fl["agent"],
           "severity": fl.get("severity", "?"), "confidence": fl.get("confidence")}
          for m in WORKING for vi, p in enumerate(papers) for r in ok_runs(m, p) for fl in r.get("flags", [])]

    # ── fig1: complete picture — score + density ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric in zip(axes, ["score", "density"]):
        means = [agg(p, metric)[0] for p in papers]; sems = [agg(p, metric)[1] for p in papers]
        bars = ax.bar(VARIANTS, means, color=VAR_COLOR, alpha=0.9, yerr=sems, capsize=6,
                      error_kw={"linewidth": 1.3}, zorder=3)
        top = max(means) or 1
        for b, mv, s in zip(bars, means, sems):
            ax.text(b.get_x()+b.get_width()/2, mv+s+top*0.04, f"{mv:.2f}", ha="center",
                    va="bottom", fontsize=10.5, fontweight="bold")
        order = " > ".join(VSHORT[i] for i in np.argsort(means)[::-1])
        ax.text(0.02, 0.97, order, transform=ax.transAxes, ha="left", va="top",
                fontsize=9, style="italic", color="#555")
        ax.set_ylabel(YLAB[metric], fontsize=10); ax.set_ylim(0, top*1.45)
        ax.set_title("Bias score (severity-weighted)" if metric == "score" else "Flag density (raw count)",
                     fontsize=11, fontweight="bold")
    fig.suptitle(f"{title} study — bias by document origin (8 detector models, mean ± SEM)",
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout(); fig.savefig(OUT/"fig1_complete_picture.png", bbox_inches="tight"); plt.close(fig)

    # ── fig2: per-model score heatmap ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    mat = np.array([[model_score(m, p) or 0 for p in papers] for m in rows])
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=max(mat.max(), 0.1), aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(VARIANTS, fontsize=9.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([lbl(m) for m in rows], fontsize=9.5)
    for i in range(len(rows)):
        for j in range(3):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if mat[i,j] > mat.max()*0.55 else "#333")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("Bias score (0–10)", fontsize=10)
    ax.set_title(f"{title} study — per-model bias score by document origin\ndarker = more bias detected",
                 fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout(); fig.savefig(OUT/"fig2_score_heatmap.png", bbox_inches="tight"); plt.close(fig)

    # ── fig3: de-biasing slope (score) ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.2)); x = np.arange(3)
    for m in rows:
        ax.plot(x, [model_score(m, p) or 0 for p in papers], "-o", lw=1.3, ms=4.5, alpha=0.55, color="#888")
    ax.plot(x, [agg(p, "score")[0] for p in papers], "-o", lw=3, ms=9, color="#C0392B", label="pooled mean", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(["Human\n(OG)", "GPT-\nGenerated", "GPT-\nEdited"], fontsize=10)
    ax.set_ylabel("Bias score (0–10)", fontsize=10.5); ax.legend(fontsize=9.5)
    ax.set_title(f"{title} study — detector tracks the editing intervention\n(grey = models, red = pooled mean)",
                 fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout(); fig.savefig(OUT/"fig3_debiasing_effect.png", bbox_inches="tight"); plt.close(fig)

    # ── fig4: radar — bias-type fingerprint by variant ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for vi, p in enumerate(papers):
        vals = [density_bt(p, bt) for bt in BIAS_TYPES]; vals += vals[:1]
        ax.plot(ang, vals, "-o", lw=2, ms=5, color=VAR_COLOR[vi], label=VSHORT[vi])
        ax.fill(ang, vals, color=VAR_COLOR[vi], alpha=0.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([BIAS_LABEL[b] for b in BIAS_TYPES], fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=9.5, title="Origin")
    ax.set_title(f"{title} study — bias-type fingerprint by origin\n(flag density / 1,000 words)",
                 fontsize=12, fontweight="bold", pad=24)
    fig.tight_layout(); fig.savefig(OUT/"fig4_radar_biastype.png", bbox_inches="tight"); plt.close(fig)

    # ── fig5: per-model radar small multiples (this study's docs) ────────────
    def prof(m):
        c = collections.Counter()
        for p in papers:
            for r in ok_runs(m, p):
                for fl in r.get("flags", []): c[fl["bias_type"]] += 1
        return [c[b] for b in BIAS_TYPES]
    vmax = max((max(prof(m)) for m in rows), default=1) or 1
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), subplot_kw=dict(polar=True))
    for ax, m in zip(axes.flat, rows):
        vals = prof(m); vals += vals[:1]
        ax.plot(ang, vals, "-o", lw=1.8, ms=4, color="#C0392B"); ax.fill(ang, vals, color="#C0392B", alpha=0.18)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels([BIAS_LABEL[b] for b in BIAS_TYPES], fontsize=8)
        ax.set_ylim(0, vmax); ax.set_yticklabels([]); ax.set_title(lbl(m), fontsize=11, fontweight="bold", pad=14)
    fig.suptitle(f"{title} study — per-model detection profile (flag counts across 3 variants)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(OUT/"fig5_radar_per_model.png", bbox_inches="tight"); plt.close(fig)

    # ── fig6: bias-type composition (normalised) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    bt_colors = plt.cm.Set2(np.linspace(0, 1, len(BIAS_TYPES)))
    comp = np.array([[density_bt(p, bt) for p in papers] for bt in BIAS_TYPES])
    tot = comp.sum(axis=0); tot[tot == 0] = 1; pct = comp/tot*100; bottoms = np.zeros(3)
    for bi, bt in enumerate(BIAS_TYPES):
        ax.bar(VARIANTS, pct[bi], bottom=bottoms, color=bt_colors[bi], label=BIAS_LABEL[bt],
               edgecolor="white", linewidth=0.6); bottoms += pct[bi]
    ax.set_ylabel("% of detected bias", fontsize=10.5); ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9.5, title="Bias type")
    ax.set_title(f"{title} study — composition of detected bias by origin", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT/"fig6_biastype_composition.png", bbox_inches="tight"); plt.close(fig)

    # ── fig7: agent firing heatmap (agent × variant) ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    amat = np.zeros((len(AGENTS), 3))
    for f in FL:
        if f["agent"] in AGENTS: amat[AGENTS.index(f["agent"]), f["variant"]] += 1
    im = ax.imshow(amat, cmap="Purples", aspect="auto", vmin=0, vmax=max(amat.max(), 1))
    ax.set_xticks(range(3)); ax.set_xticklabels(VARIANTS, fontsize=9.5)
    ax.set_yticks(range(len(AGENTS))); ax.set_yticklabels(AGENTS, fontsize=10)
    for i in range(len(AGENTS)):
        for j in range(3):
            ax.text(j, i, f"{int(amat[i,j])}", ha="center", va="center", fontsize=10,
                    color="white" if amat[i,j] > amat.max()*0.55 else "#333")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("Flags raised", fontsize=10)
    ax.set_title(f"{title} study — which agent raised the flag", fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout(); fig.savefig(OUT/"fig7_agent_firing.png", bbox_inches="tight"); plt.close(fig)

    # ── fig8: severity mix ───────────────────────────────────────────────────
    SEV = ["low", "medium", "high"]; SEV_COLOR = ["#A8D5BA", "#F2C14E", "#E07A5F"]
    fig, ax = plt.subplots(figsize=(7, 5)); counts = np.zeros((3, 3))
    for f in FL:
        if f["severity"] in SEV: counts[SEV.index(f["severity"]), f["variant"]] += 1
    tot = counts.sum(axis=0); tot[tot == 0] = 1; pct = counts/tot*100; bottoms = np.zeros(3)
    for si, s in enumerate(SEV):
        ax.bar(VARIANTS, pct[si], bottom=bottoms, color=SEV_COLOR[si], label=s.capitalize(), edgecolor="white")
        bottoms += pct[si]
    ax.set_ylabel("% of flags", fontsize=10.5); ax.set_ylim(0, 100); ax.legend(title="Severity", fontsize=9.5)
    ax.set_title(f"{title} study — severity mix by origin", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT/"fig8_severity_mix.png", bbox_inches="tight"); plt.close(fig)

    # ── fig9: confidence distribution ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [[f["confidence"] for f in FL if f["variant"] == vi and f["confidence"] is not None] or [np.nan]
              for vi in range(3)]
    bp = ax.boxplot(groups, positions=range(3), widths=0.6, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], VAR_COLOR): patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticks(range(3)); ax.set_xticklabels(VARIANTS, fontsize=9.5)
    ax.set_ylabel("Flag confidence", fontsize=10.5)
    ax.set_title(f"{title} study — confidence of raised flags", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT/"fig9_confidence.png", bbox_inches="tight"); plt.close(fig)

    # ── fig10: score vs density scatter ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for vi, p in enumerate(papers):
        for m in WORKING:
            sc, de = model_score(m, p), model_density(m, p)
            if sc is None: continue
            ax.scatter(de, sc, s=70, color=VAR_COLOR[vi], alpha=0.75, edgecolor="white", linewidth=0.5)
    leg = [Line2D([0],[0], marker="o", color="w", markerfacecolor=VAR_COLOR[i], markersize=10, label=VSHORT[i])
           for i in range(3)]
    ax.legend(handles=leg, fontsize=9, loc="upper left")
    ax.set_xlabel("Flag density (flags / 1,000 words)"); ax.set_ylabel("Bias score (0–10)")
    ax.set_title(f"{title} study — score vs. flag density", fontsize=12, fontweight="bold"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT/"fig10_score_vs_density.png", bbox_inches="tight"); plt.close(fig)

    # ── fig11: run-to-run consistency (SD of density) ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5)); sds = []
    for m in rows:
        ps = []
        for p in papers:
            d = [r.get("n_flags", 0)*1000/DOC_WORDS[p] for r in ok_runs(m, p)]
            if len(d) >= 2: ps.append(np.std(d, ddof=1))
        sds.append(np.mean(ps) if ps else 0)
    bars = ax.bar([lbl(m) for m in rows], sds, color="#6A8CAF", alpha=0.88, zorder=3)
    for b, v in zip(bars, sds):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean within-doc SD of flag density\n(lower = more consistent)", fontsize=10)
    ax.set_title(f"{title} study — run-to-run consistency by model", fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=22, ha="right")
    fig.tight_layout(); fig.savefig(OUT/"fig11_consistency.png", bbox_inches="tight"); plt.close(fig)

    # ── fig12: ranked model leaderboard (mean score across 3 variants) ───────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rank = sorted(((lbl(m), np.mean([model_score(m, p) or 0 for p in papers])) for m in WORKING), key=lambda x: x[1])
    names = [r[0] for r in rank]; vals = [r[1] for r in rank]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
    bars = ax.barh(names, vals, color=cmap, alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(v+0.01, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean bias score across 3 document variants (0–10)")
    ax.set_title(f"{title} study — detector sensitivity leaderboard", fontsize=12, fontweight="bold", pad=8)
    ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT/"fig12_model_leaderboard.png", bbox_inches="tight"); plt.close(fig)

    print(f"{title}: 12 figures -> {OUT}")


if __name__ == "__main__":
    for key in STUDIES:
        render(key)
    print(f"\nDone. Working detectors: {len(WORKING)}/{len(data['results'])}")
