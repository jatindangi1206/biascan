"""Shared document-level metric recomputation for the leaderboard analyses.

Single source of truth so the figures (leaderboard_scale_viz) and the
significance test (edit_significance) compute coverage / impact / score the
same way. Definitions follow docs/SCORING.md (the stored `score` field in
leaderboard.json predates that formula and is NOT used here):

  impact   = Σ confidence × severity_weight  +  0.15 × (unique_bias_types − 1)
             severity weight: high=3, medium=2, low=1
  coverage = union of flagged words / total words  (spans located by
             substring-matching each flag's flagged_text in the source .txt)
  score    = min(10, impact × coverage × 3.33)     (0–10 scale)
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SYN  = ROOT / "backend" / "app" / "synthesis"

STUDIES = {
    "MS-Gut Review":         ["MS_GUT_OG", "MS_GUT_GPT_Made", "MS_GUT_GPT_Edit"],
    "Nutraceuticals Review": ["Nu_OG",     "Nu_GPT_OG",       "Nu_GPT_Edit"],
}
VARIANTS = ["Human Written (Original)", "LLM Written", "LLM Edited"]
COLORS   = ["#3B6BA5", "#E1812C", "#4F9D69"]
SEVW     = {"high": 3.0, "medium": 2.0, "low": 1.0}

DOCS = {p: (SYN / f"{p}.txt").read_text() for papers in STUDIES.values() for p in papers}
DOC_WORDS = {p: len(t.split()) for p, t in DOCS.items()}


def working_models(data, min_flags: int = 10):
    """Models whose detectors actually fired (excludes parse-failure models)."""
    return [r["model"] for r in data["results"]
            if sum(run.get("n_flags", 0) for pv in r["papers"].values()
                   for run in pv.get("runs", [])) >= min_flags]


def ok_runs(res, m, p):
    return [run for run in res[m]["papers"].get(p, {}).get("runs", []) if run.get("ok")]


def run_coverage(run, paper):
    text = DOCS[paper]
    spans = [(mm.start(), mm.end()) for mm in re.finditer(r"\S+", text)]
    if not spans:
        return 0.0
    covered = [False] * len(spans)
    for f in run.get("flags", []):
        ft = (f.get("flagged_text") or "").strip()
        idx = text.find(ft) if ft else -1
        if idx < 0:
            continue
        a, b = idx, idx + len(ft)
        for i, (ws, we) in enumerate(spans):
            if ws < b and we > a:
                covered[i] = True
    return sum(covered) / len(spans)


def run_impact(run):
    flags = run.get("flags", [])
    imp = sum(SEVW.get(f.get("severity", "low"), 1.0) * (f.get("confidence") or 0) for f in flags)
    ut = len({f["bias_type"] for f in flags})
    return imp + 0.15 * max(0, ut - 1)


def run_density(run, paper):
    return run.get("n_flags", 0) * 1000 / DOC_WORDS[paper]


def run_score(run, paper):
    return min(10.0, run_impact(run) * run_coverage(run, paper) * 3.33)


def run_conf(run):
    cs = [f.get("confidence") for f in run.get("flags", []) if f.get("confidence") is not None]
    return float(np.mean(cs)) if cs else 0.0
