"""AEGIS ablation — validate the detector agents independently of AEGIS.

Question this answers
---------------------
Are the five specialist agents (ARGUS/LIBRA/LENS/QUILL/VIGIL) actually
detecting the right bias, or is a bad final label AEGIS's fault?

We run each labeled sentence through the agents *directly* and ask a single
yes/no question per sentence: **did the correct specialist agent fire?**
Overlap is irrelevant here — if ARGUS *and* QUILL both flag a confirmation-bias
sentence, that's still a hit for ARGUS. We only care whether the correct agent
is among those that fired.

  - High agent recall  → the detectors work. Any wrong final label the product
    emits is then AEGIS's conflict-resolution reasoning, not the agents.
  - Low agent recall   → the problem is upstream, in the agents themselves.

Why this does NOT touch the scoring pipeline
--------------------------------------------
This script calls ``agent.run()`` directly. It never builds the Orchestrator's
AEGIS step and never calls ``_overall_score`` / ``_doc_metrics``. There is no
score computed anywhere, so the "overlaps get double-counted" problem simply
cannot occur — it's a set-membership check, fully separate from the shipped
scoring path. No shipped code is modified; the agents are imported read-only.

Run
---
  export BIASSCAN_API_KEY=sk-or-...        # OpenRouter key
  python -m eval.aegis_ablation                      # systematic_review_v1 suite
  python -m eval.aegis_ablation --runs 5 --concurrency 5
  python -m eval.aegis_ablation --selfcheck          # offline logic check, no network

Writes eval/output/aegis_ablation_per_run.csv and prints a summary.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agents import ALL_AGENTS  # noqa: E402
from app.agents.orchestrator import Orchestrator  # noqa: E402
from app.config import CONFIDENCE_FLOOR  # noqa: E402
from app.providers import build_provider  # noqa: E402
from app.providers.base import ProviderConfig  # noqa: E402

# Reuse the sentence-eval dataset loader, suites, and model list verbatim.
from eval.sentence_eval import MODELS, SUITES, load_dataset, short  # noqa: E402

OUT_DIR = ROOT / "eval" / "output"
PER_RUN_CSV = OUT_DIR / "aegis_ablation_per_run.csv"

# Decent-tier detector chosen for the ablation: fast, cheap, and 2nd-best
# attribution accuracy in the sentence eval (42%, 80% detection). We run the
# ablation on this one model rather than re-running all five.
# Override with --model, or run every model with --all.
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEGIS ablation — agent-level recall, no AEGIS, no scoring.")
    p.add_argument("--suite", choices=sorted(SUITES.keys()), default="systematic_review_v1",
                   help="Which labeled suite to run (default: the product suite).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"OpenRouter model id to run (default: {DEFAULT_MODEL}).")
    p.add_argument("--all", action="store_true",
                   help="Run every model in sentence_eval.MODELS instead of a single one.")
    p.add_argument("--runs", type=int, default=3, help="Repeats per (sentence, model). 5 matches sentence_eval.")
    p.add_argument("--concurrency", type=int, default=3, help="Max in-flight provider calls (raise if not rate-limited).")
    p.add_argument("--with-aegis", action="store_true",
                   help="Run the FULL pipeline (agents + AEGIS) and measure POST-AEGIS attribution "
                        "accuracy instead of pre-AEGIS agent recall. Use this to test AEGIS changes.")
    p.add_argument("--selfcheck", action="store_true", help="Run the offline aggregation self-check and exit.")
    return p.parse_args()


# ── Core measurement (one provider call per agent, no AEGIS, no score) ────────

async def _run_agent(agent, sentence, provider, analysis_mode, sem) -> tuple[str, list, str | None]:
    async with sem:
        anns, err, _ = await agent.run(
            text=sentence, source_text=sentence, references=None,
            mode="lite", analysis_mode=analysis_mode, provider=provider,
        )
    kept = [a for a in anns if a.confidence >= CONFIDENCE_FLOOR]
    return agent.name, kept, err


async def measure_sentence(agents, sentence, provider, analysis_mode, sem) -> dict:
    """Fire all agents on one sentence. Returns which agents fired + their labels."""
    results = await asyncio.gather(
        *(_run_agent(a, sentence, provider, analysis_mode, sem) for a in agents)
    )
    fired = {name for name, kept, _ in results if kept}
    errors = sum(1 for _, _, err in results if err)
    return {"fired_agents": fired, "n_fired": len(fired), "errors": errors}


async def measure_sentence_full(orch, sentence, cfg, analysis_mode, sem) -> dict:
    """Full pipeline (agents + AEGIS). Returns the POST-AEGIS bias types on the span."""
    async with sem:
        resp = await orch.analyze(
            text=sentence, references=None, mode="lite",
            analysis_mode=analysis_mode, provider_config=cfg, agents=None,
        )
    detected = {a.bias_type for a in resp.annotations}
    return {"fired_agents": detected, "n_fired": len(detected), "errors": 0}


# ── Aggregation (pure, offline-testable) ─────────────────────────────────────

def summarize(records: list[dict], agent_names: list[str]) -> dict:
    """records: one dict per (model, sentence, run) with keys
    expected_agent, fired_agents (set), n_fired. Returns summary metrics."""
    n = len(records)
    if n == 0:
        return {"n": 0}

    hits = sum(r["expected_agent"] in r["fired_agents"] for r in records)

    # Recall per expected agent (= per bias type, since each agent owns one type)
    per_agent_hit = defaultdict(list)
    for r in records:
        per_agent_hit[r["expected_agent"]].append(int(r["expected_agent"] in r["fired_agents"]))
    per_agent_recall = {a: (sum(v) / len(v), len(v)) for a, v in per_agent_hit.items()}

    # Per-sentence firing rate — the sentence (per model) is the sampling unit,
    # exactly like the score eval. A sentence whose correct agent fires 2 of 5
    # runs has rate 0.40; we keep it, we don't discard it.
    by_sentence = defaultdict(list)
    for r in records:
        by_sentence[(r["model"], r["sentence_id"])].append(int(r["expected_agent"] in r["fired_agents"]))
    sent_rates = [sum(v) / len(v) for v in by_sentence.values()]
    n_sent = len(sent_rates)
    sent_mean = statistics.mean(sent_rates) if sent_rates else 0.0
    sent_sd = statistics.stdev(sent_rates) if n_sent > 1 else 0.0
    buckets = {
        "always (5/5)":   sum(x == 1.0 for x in sent_rates),
        "majority (>=3/5)": sum(0.5 <= x < 1.0 for x in sent_rates),
        "flaky (1-2/5)":  sum(0.0 < x < 0.5 for x in sent_rates),
        "never (0/5)":    sum(x == 0.0 for x in sent_rates),
    }

    # Overlap / multiplicity — how often AEGIS would even be invoked
    multi = sum(r["n_fired"] >= 2 for r in records)
    mean_fired = sum(r["n_fired"] for r in records) / n

    # Off-target firing: when a NON-expected agent fires, tally it (over-firing tendency)
    off_target = defaultdict(int)
    for r in records:
        for a in r["fired_agents"]:
            if a != r["expected_agent"]:
                off_target[a] += 1

    return {
        "n": n,
        "recall": hits / n,                 # pooled per-run recall
        "per_agent_recall": per_agent_recall,
        "n_sentences": n_sent,
        "sent_mean": sent_mean,             # mean per-sentence firing rate
        "sent_sd": sent_sd,                 # SD across sentences (consistency)
        "buckets": buckets,
        "p_multi": multi / n,
        "mean_fired": mean_fired,
        "off_target": dict(off_target),
    }


def print_summary(s: dict, agent_by_bias: dict[str, str], post_aegis: bool = False) -> None:
    if s.get("n", 0) == 0:
        print("No records."); return
    bias_by_agent = {v: k for k, v in agent_by_bias.items()}
    title = ("AEGIS TEST — full pipeline, POST-AEGIS attribution accuracy" if post_aegis
             else "AEGIS ABLATION — detector agents measured directly (AEGIS off, no score)")
    metric = "Attribution accuracy" if post_aegis else "Agent recall"
    unit = ("fraction of runs where the correct bias TYPE survived AEGIS" if post_aegis
            else "fraction of runs where the CORRECT specialist agent fired")
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Records (model × sentence × run): {s['n']}")
    print(f"\n>> HEADLINE  {metric} = {s['recall']*100:.1f}%  (pooled over every run)")
    print(f"   ({unit})")

    print(f"\nPer-sentence view (sentence = sampling unit, n={s['n_sentences']}):")
    print(f"  mean rate = {s['sent_mean']*100:.1f}%  ±{s['sent_sd']*100:.1f}%  (SD across sentences)")
    for label, c in s["buckets"].items():
        print(f"    {label:<18} {c} sentences")
    print("  (a sentence surviving 2/5 = rate 0.40 → counted in 'flaky', not discarded)\n")

    print(f"Per-bias-type {'accuracy' if post_aegis else 'recall'}:")
    for key, (rec, k) in sorted(s["per_agent_recall"].items(), key=lambda x: -x[1][0]):
        # post: key is the bias type; pre: key is the agent name
        bias = key if post_aegis else bias_by_agent.get(key, "?")
        agent = agent_by_bias.get(key, "?") if post_aegis else key
        print(f"  {agent:6} ({bias:24}) {rec*100:5.1f}%   (n={k})")

    if post_aegis:
        print(f"\nFinal-output shape:")
        print(f"  mean bias types / sentence : {s['mean_fired']:.2f}")
        print(f"  sentences with >=2 types   : {s['p_multi']*100:.1f}%")
    else:
        print(f"\nOverlap — how often AEGIS is actually needed:")
        print(f"  mean agents firing / sentence : {s['mean_fired']:.2f}")
        print(f"  sentences with >=2 agents     : {s['p_multi']*100:.1f}%  (the conflicts AEGIS resolves)")

    if s["off_target"]:
        noun = "wrong bias type in final output" if post_aegis else "a non-owning agent fired"
        print(f"\nOff-target ({noun}):")
        for key, c in sorted(s["off_target"].items(), key=lambda x: -x[1]):
            print(f"  {key:24} {c} times")

    print("\nRead-out:")
    if post_aegis:
        print("  Compare against pre-AEGIS agent recall (ablation ≈ 79.6%) and the")
        print("  baseline post-AEGIS accuracy (≈ 42.1%). Higher = the AEGIS change helped.")
    else:
        print("  If recall is HIGH but the shipped (post-AEGIS) accuracy is LOW,")
        print("  the detectors are fine and AEGIS's conflict resolution is the culprit.")
    print("=" * 70)


# ── Driver ───────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> int:
    api_key = os.environ.get("BIASSCAN_API_KEY")
    if not api_key:
        print("ERROR: set BIASSCAN_API_KEY (OpenRouter key) in the environment.", file=sys.stderr)
        return 2

    spec = SUITES[args.suite]
    analysis_mode = spec["analysis_mode"]
    agents = [cls() for cls in ALL_AGENTS]
    agent_by_bias = {a.bias_type: a.name for a in agents}  # derived, not hardcoded

    models = MODELS if args.all else [
        next((m for m in MODELS if m["id"] == args.model), {"id": args.model, "company": "?"})
    ]

    rows = load_dataset(args.suite)
    print(f"Suite: {args.suite} · analysis_mode={analysis_mode} · {len(rows)} sentences")
    print(f"Models: {', '.join(m['id'] for m in models)}")
    print(f"Plan: {len(rows)} × {len(models)} model(s) × {args.runs} runs × {len(agents)} agents = "
          f"{len(rows)*len(models)*args.runs*len(agents)} provider calls\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    use_aegis = args.with_aegis
    out_csv = OUT_DIR / ("aegis_ablation_full_per_run.csv" if use_aegis else "aegis_ablation_per_run.csv")
    orch = Orchestrator() if use_aegis else None
    fired_label = "POST-AEGIS label survived" if use_aegis else "correct-agent fired"

    cols = ["suite", "model", "sentence_id", "expected_bias_type", "expected_agent",
            "run", "fired_agents", "n_fired", "expected_fired", "errors"]
    out_csv.write_text(",".join(cols) + "\n")

    sem = asyncio.Semaphore(args.concurrency)
    records: list[dict] = []
    started = time.time()

    for mi, mm in enumerate(models, 1):
        cfg = ProviderConfig(provider="openrouter", model=mm["id"], api_key=api_key, base_url=None)
        provider = None if use_aegis else build_provider(cfg)
        print(f"[{mi}/{len(models)}] {mm['company']:<10} {mm['id']}"
              f"{'  (+AEGIS, full pipeline)' if use_aegis else ''}", flush=True)
        for si, row in enumerate(rows, 1):
            # AEGIS mode: 'expected' is the bias TYPE and 'fired' holds post-AEGIS
            # bias types. Ablation mode: 'expected' is the agent NAME, 'fired' the agents.
            expected_key = (row["expected_bias_type"] if use_aegis
                            else agent_by_bias[row["expected_bias_type"]])
            if use_aegis:
                run_measures = await asyncio.gather(
                    *(measure_sentence_full(orch, row["sentence"], cfg, analysis_mode, sem)
                      for _ in range(args.runs)))
            else:
                run_measures = await asyncio.gather(
                    *(measure_sentence(agents, row["sentence"], provider, analysis_mode, sem)
                      for _ in range(args.runs)))
            hits = sum(expected_key in mres["fired_agents"] for mres in run_measures)
            print(f"  [{si:>2}/{len(rows)}] id={row['sentence_id']:<4} exp={expected_key:<22} "
                  f"{fired_label} {hits}/{args.runs}", flush=True)
            for ri, mres in enumerate(run_measures, 1):
                fired = mres["fired_agents"]
                rec = {
                    "suite": args.suite, "model": mm["id"],
                    "sentence_id": row["sentence_id"],
                    "expected_bias_type": row["expected_bias_type"],
                    "expected_agent": expected_key,
                    "run": ri,
                    "fired_agents": "|".join(sorted(fired)),
                    "n_fired": mres["n_fired"],
                    "expected_fired": int(expected_key in fired),
                    "errors": mres["errors"],
                }
                records.append({**rec, "fired_agents": fired})  # in-memory keeps the set
                with out_csv.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=cols, extrasaction="ignore").writerow(rec)

    print(f"\nElapsed {time.time()-started:.0f}s · wrote {out_csv}")
    print_summary(summarize(records, [a.name for a in agents]), agent_by_bias, post_aegis=use_aegis)
    return 0


# ── Offline self-check (no network) ──────────────────────────────────────────

def selfcheck() -> int:
    agents = ["ARGUS", "LIBRA", "LENS", "QUILL", "VIGIL"]
    m = "model-x"
    recs = []
    # sA: ARGUS fires 5/5 (always); overlaps QUILL once
    for i in range(5):
        fired = {"ARGUS", "QUILL"} if i == 0 else {"ARGUS"}
        recs.append({"model": m, "sentence_id": "sA", "expected_agent": "ARGUS",
                     "fired_agents": fired, "n_fired": len(fired)})
    # sB: LIBRA fires 2/5 (the "2 of 5" flaky case); other runs fire QUILL off-target
    for i in range(5):
        fired = {"LIBRA"} if i < 2 else {"QUILL"}
        recs.append({"model": m, "sentence_id": "sB", "expected_agent": "LIBRA",
                     "fired_agents": fired, "n_fired": 1})
    # sC: LENS never fires (0/5)
    for i in range(5):
        recs.append({"model": m, "sentence_id": "sC", "expected_agent": "LENS",
                     "fired_agents": set(), "n_fired": 0})
    s = summarize(recs, agents)
    assert s["n"] == 15
    assert abs(s["recall"] - 7/15) < 1e-9, s["recall"]               # (5+2+0)/15 pooled
    assert s["n_sentences"] == 3
    assert abs(s["sent_mean"] - (1.0 + 0.4 + 0.0) / 3) < 1e-9, s["sent_mean"]
    assert s["buckets"] == {"always (5/5)": 1, "majority (>=3/5)": 0,
                            "flaky (1-2/5)": 1, "never (0/5)": 1}, s["buckets"]
    assert s["per_agent_recall"]["ARGUS"][0] == 1.0
    assert abs(s["per_agent_recall"]["LIBRA"][0] - 0.4) < 1e-9
    assert s["off_target"]["QUILL"] == 1 + 3                         # 1 on sA + 3 on sB
    print("selfcheck OK — pooled recall, per-sentence mean/SD, buckets, off-target all correct")
    return 0


if __name__ == "__main__":
    a = parse_args()
    if a.selfcheck:
        sys.exit(selfcheck())
    sys.exit(asyncio.run(main(a)))
