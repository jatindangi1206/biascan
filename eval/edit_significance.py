"""Significance test: does the LLM editing pipeline reduce detected bias?

Primary comparison: LLM Written vs LLM Edited, per study (MS-GUT, Nu).

Metrics are recomputed from the stored flags using the docs/SCORING.md
definitions (see eval/_doc_metrics.py) — bias score (primary), span coverage,
and severity-weighted impact. The stored `score` field predates that formula
and is not used.

DESIGN
------
Unit of analysis = MODEL (paired). Each model's value is the mean over its 5
"ok" runs; the 5 runs are repeated measures, so the model is the sampling
unit. We pair LLM-Written vs LLM-Edited within each model and test the n
paired differences d_i = Written_i - Edited_i (positive => editing reduced
the detected bias). Only the working detectors (>=10 flags total) are used,
matching the figures and the protocol's exclusion of parse-failure models.

TESTS (exact, dependency-free — scipy is unavailable in this env, and exact
enumeration is in fact stronger than the normal approximation at n=8)
--------------------------------------------------------------------------
1. Paired sign-flip permutation test (two-sided).
   Under H0 each d_i is equally likely + or -. Enumerate all 2^n sign vectors
   s in {-1,+1}^n; statistic T_s = sum(s_i * |d_i|); observed T = sum(d_i).
   p = #{ s : |T_s| >= |T_obs| } / 2^n.

2. Wilcoxon signed-rank, exact (two-sided).
   Drop zero diffs (m left). Rank |d| (average ranks for ties). W+ = sum of
   ranks of positive diffs. Enumerate all 2^m sign assignments of the ranks to
   build the exact null distribution of W+; p = P(|W+ - E| >= |obs - E|),
   E = sum(ranks)/2.

3. Effect size: paired Cohen's d_z = mean(d) / sd(d). Paired t = d_z * sqrt(n)
   (statistic reported; p comes from the exact tests above).

Run:  python -m eval.edit_significance
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path

import numpy as np

from eval._doc_metrics import (
    working_models, ok_runs, run_score, run_coverage, run_impact,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = json.load((ROOT / "eval" / "output" / "leaderboard.json").open())
RES = {r["model"]: r for r in DATA["results"]}
WORKING = working_models(DATA)

# Written vs Edited paper per study
PAIRS = {
    "MS-Gut Review":         ("MS_GUT_GPT_Made", "MS_GUT_GPT_Edit"),
    "Nutraceuticals Review": ("Nu_GPT_OG",       "Nu_GPT_Edit"),
}


def model_mean(m, paper, fn):
    runs = ok_runs(RES, m, paper)
    if not runs:
        return None
    return float(np.mean([fn(r, paper) if fn.__code__.co_argcount == 2 else fn(r) for r in runs]))


def _avg_rank(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def perm_p(diffs):
    a = np.abs(diffs); obs = abs(diffs.sum()); n = len(diffs)
    hits = sum(1 for s in itertools.product((-1, 1), repeat=n)
               if abs(np.dot(s, a)) >= obs - 1e-12)
    return hits / 2 ** n


def wilcoxon_exact(diffs):
    d = diffs[diffs != 0]; m = len(d)
    if m == 0:
        return 0.0, 1.0
    ranks = _avg_rank(np.abs(d))
    wplus = float(ranks[d > 0].sum()); expect = ranks.sum() / 2; dev = abs(wplus - expect)
    hits = sum(1 for s in itertools.product((0, 1), repeat=m)
               if abs(np.dot(s, ranks) - expect) >= dev - 1e-12)
    return wplus, hits / 2 ** m


def report(metric_name, fn):
    print(f"\n{'='*74}\nMETRIC: {metric_name}   (unit = model, n={len(WORKING)} working detectors, paired)\n{'='*74}")
    for study, (wp, ep) in PAIRS.items():
        w = np.array([model_mean(m, wp, fn) for m in WORKING], float)
        e = np.array([model_mean(m, ep, fn) for m in WORKING], float)
        mask = ~(np.isnan(w) | np.isnan(e)); w, e = w[mask], e[mask]
        diff = w - e                      # + => Written more biased (editing reduced it)
        n = len(diff)
        mw, me, md = w.mean(), e.mean(), diff.mean()
        sd = diff.std(ddof=1) if n > 1 else 0.0
        dz = md / sd if sd > 0 else float("nan")
        t = dz * np.sqrt(n) if sd > 0 else float("nan")
        pct = 100 * (mw - me) / mw if mw > 0 else float("nan")
        wplus, p_w = wilcoxon_exact(diff); p_p = perm_p(diff)
        npos, nneg, ntie = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
        print(f"\n{study}:  Written={wp}   Edited={ep}")
        print(f"  mean Written = {mw:.3f}   mean Edited = {me:.3f}   mean drop = {md:+.3f}  ({pct:.0f}% reduction)")
        print(f"  per-model pairs: {npos} Written>Edited, {nneg} Written<Edited, {ntie} tie")
        print(f"  Cohen's d_z = {dz:.2f}   paired t({n-1}) = {t:.2f}")
        print(f"  Wilcoxon W+ = {wplus:.1f}   exact two-sided p = {p_w:.4f}")
        print(f"  permutation  exact two-sided p = {p_p:.4f}")


if __name__ == "__main__":
    print("Working detectors (n=%d): %s" % (len(WORKING), ", ".join(m.split('/')[-1] for m in WORKING)))
    report("Bias score (0-10, SCORING.md formula — PRIMARY)", run_score)
    report("Span coverage (fraction of words flagged)", run_coverage)
    report("Severity-weighted impact", run_impact)
