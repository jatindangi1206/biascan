"""Sentence-level agent validation.

Runs BiasScan over one named evaluation suite at a time.

Supported suites:
  - legacy: the existing sentence dataset
  - systematic_review_v1: frozen specialist systematic-review prompts
  - general_research_v2: broader general-research prompts

For each labeled example:
  - Run BiasScan analyze() N times per (example, model) pair
  - Across 5 models (one per company)
  - Record per-run results + aggregated summary
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.orchestrator import Orchestrator  # noqa: E402
from app.providers.base import ProviderConfig  # noqa: E402

N_RUNS = 5
MODELS: list[dict] = [
    {"id": "openai/gpt-4.1-nano",                  "company": "OpenAI"},
    {"id": "anthropic/claude-3-haiku",             "company": "Anthropic"},
    {"id": "google/gemini-2.5-flash-lite",         "company": "Google"},
    {"id": "deepseek/deepseek-v4-flash",           "company": "DeepSeek"},
    {"id": "meta-llama/llama-3.3-70b-instruct",    "company": "Meta"},
]

# Map the dataset's bias-label strings to our internal bias_type literals
LABEL_MAP = {
    "Confirmation Bias":   "confirmation_bias",
    "Certainty Inflation": "certainty_inflation",
    "Framing Effects":     "framing_effect",
    "Causal Inference":    "causal_inference_error",
    "Overgeneralization":  "overgeneralisation",
}

SUITES = {
    "legacy": {
        "dataset": ROOT / "backend" / "app" / "sentences-dataset" / "dataset .csv",
        "analysis_mode": "general_research",
        "id_col": "Sentence ID",
        "text_col": "Sentence Text",
        "label_col": "Annotated Bias",
    },
    "systematic_review_v1": {
        "dataset": ROOT / "eval" / "datasets" / "suites" / "systematic_review_v1.csv",
        "analysis_mode": "systematic_review",
        "id_col": "Item ID",
        "text_col": "Text",
        "label_col": "Annotated Bias",
    },
    "general_research_v2": {
        "dataset": ROOT / "eval" / "datasets" / "suites" / "general_research_v2.csv",
        "analysis_mode": "general_research",
        "id_col": "Item ID",
        "text_col": "Text",
        "label_col": "Annotated Bias",
    },
}

OUT_DIR = ROOT / "eval" / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BiasScan sentence-level evaluation.")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES.keys()),
        default="general_research_v2",
        help="Evaluation suite to run.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=N_RUNS,
        help="Number of repeated runs per (example, model) pair.",
    )
    return parser.parse_args()


def output_paths(suite: str) -> tuple[Path, Path, Path]:
    if suite == "legacy":
        return (
            OUT_DIR / "sentence_eval_per_run.csv",
            OUT_DIR / "sentence_eval_summary.csv",
            OUT_DIR / "sentence_eval_progress.json",
        )
    return (
        OUT_DIR / f"sentence_eval_per_run_{suite}.csv",
        OUT_DIR / f"sentence_eval_summary_{suite}.csv",
        OUT_DIR / f"sentence_eval_progress_{suite}.json",
    )


def load_dataset(suite: str) -> list[dict]:
    spec = SUITES[suite]
    rows = []
    with spec["dataset"].open() as f:
        for row in csv.DictReader(f):
            label = row[spec["label_col"]].strip()
            expected = LABEL_MAP.get(label)
            if expected is None:
                print(f"  ! unknown label {label!r} for row {row[spec['id_col']]}, skipping")
                continue
            rows.append({
                "sentence_id": row[spec["id_col"]].strip(),
                "sentence": row[spec["text_col"]].strip(),
                "expected_label_human": label,
                "expected_bias_type": expected,
                "text_type": (row.get("Text Type") or "").strip(),
                "why_not_neighbors": (row.get("Why Not Neighbors") or "").strip(),
                "multi_label_notes": (row.get("Multi Label Notes") or "").strip(),
            })
    return rows


def short(s: str, n: int = 220) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


async def one_run(
    orch: Orchestrator,
    sentence: str,
    cfg: ProviderConfig,
    analysis_mode: str,
) -> dict:
    """Run BiasScan on one sentence and extract eval fields."""
    try:
        resp = await orch.analyze(
            text=sentence,
            references=None,
            mode="lite",
            analysis_mode=analysis_mode,  # type: ignore[arg-type]
            provider_config=cfg,
            agents=None,
        )
        anns = resp.annotations
        agents = sorted({a.agent_name for a in anns})
        bias_types = sorted({a.bias_type for a in anns})
        severities = [a.severity for a in anns]
        confidences = [round(a.confidence, 2) for a in anns]
        # Representative reasoning: the chain_of_thought attached to the
        # highest-confidence annotation, or any agent's reasoning if no annotations.
        reasoning = ""
        if anns:
            top = max(anns, key=lambda a: a.confidence)
            cot = top.extras.get("chain_of_thought") if top.extras else None
            if isinstance(cot, dict):
                reasoning = json.dumps(cot)[:300]
            elif isinstance(cot, str):
                reasoning = cot[:300]
        if not reasoning:
            for info in resp.agents:
                if info.reasoning and isinstance(info.reasoning, dict):
                    reasoning = json.dumps(info.reasoning)[:300]
                    break
        # Error warnings count — quick health signal
        errs = sum(1 for w in resp.warnings if "openrouter error" in w.lower() or "error 4" in w.lower())
        return {
            "ok": True,
            "score": round(resp.overall_bias_score * 10, 2),
            "n_flags": len(anns),
            "agents_fired": "|".join(agents),
            "bias_types_detected": "|".join(bias_types),
            "severities": "|".join(severities),
            "confidences": "|".join(str(c) for c in confidences),
            "reasoning": short(reasoning),
            "api_errors": errs,
        }
    except Exception as e:
        return {"ok": False, "score": None, "n_flags": 0, "agents_fired": "",
                "bias_types_detected": "", "severities": "", "confidences": "",
                "reasoning": "", "api_errors": 1, "error": f"{type(e).__name__}: {e}"}


async def main(args: argparse.Namespace) -> int:
    api_key = os.environ["BIASSCAN_API_KEY"]
    spec = SUITES[args.suite]
    per_run_csv, summary_csv, progress_json = output_paths(args.suite)
    rows = load_dataset(args.suite)
    print(f"Loaded {len(rows)} examples from {spec['dataset']}\n")
    print(
        f"Suite: {args.suite} · analysis_mode={spec['analysis_mode']}\n"
        f"Plan: {len(rows)} examples × {len(MODELS)} models × {args.runs} runs = "
        f"{len(rows) * len(MODELS) * args.runs} analyze() calls\n"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    orch = Orchestrator()

    # Per-run rows accumulated in memory + incrementally flushed to CSV.
    per_run_rows: list[dict] = []
    started = time.time()

    # Schema for per-run CSV
    per_run_cols = [
        "suite", "analysis_mode",
        "sentence_id", "sentence", "expected_label_human", "expected_bias_type",
        "text_type", "why_not_neighbors", "multi_label_notes",
        "company", "model", "run",
        "score", "n_flags", "agents_fired", "bias_types_detected",
        "severities", "confidences", "correct", "reasoning",
        "api_errors", "ok", "error",
    ]
    per_run_csv.write_text(",".join(per_run_cols) + "\n")

    def append_per_run(d: dict) -> None:
        per_run_rows.append(d)
        with per_run_csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_run_cols, extrasaction="ignore")
            w.writerow(d)

    for m_idx, mm in enumerate(MODELS, 1):
        model = mm["id"]
        cfg = ProviderConfig(provider="openrouter", model=model, api_key=api_key, base_url=None)
        print(f"\n{'#'*80}\n# [{m_idx}/{len(MODELS)}] {mm['company']:<10} · {model}\n{'#'*80}",
              flush=True)
        for s_idx, row in enumerate(rows, 1):
            sentence = row["sentence"]
            # Run all N_RUNS in parallel for one (sentence, model) pair
            run_results = await asyncio.gather(
                *(one_run(orch, sentence, cfg, spec["analysis_mode"]) for _ in range(args.runs))
            )
            scores = [r["score"] for r in run_results if r["ok"] and r["score"] is not None]
            mean = round(statistics.mean(scores), 2) if scores else None
            sd = round(statistics.stdev(scores), 2) if len(scores) >= 2 else 0.0
            print(f"  [{s_idx:>2}/{len(rows)}] id={row['sentence_id']:<3} "
                  f"({row['expected_bias_type'][:18]:<18}) "
                  f"runs=[{','.join(str(r['score']) for r in run_results)}] "
                  f"mean={mean} sd={sd}", flush=True)
            for run_idx, rr in enumerate(run_results, 1):
                # "correct" = expected bias_type appears in detected set
                detected_set = set(rr.get("bias_types_detected", "").split("|"))
                detected_set.discard("")
                correct = row["expected_bias_type"] in detected_set
                append_per_run({
                    "suite": args.suite,
                    "analysis_mode": spec["analysis_mode"],
                    "sentence_id": row["sentence_id"],
                    "sentence": short(row["sentence"], 280),
                    "expected_label_human": row["expected_label_human"],
                    "expected_bias_type": row["expected_bias_type"],
                    "text_type": row["text_type"],
                    "why_not_neighbors": row["why_not_neighbors"],
                    "multi_label_notes": row["multi_label_notes"],
                    "company": mm["company"],
                    "model": model,
                    "run": run_idx,
                    "score": rr.get("score") if rr["ok"] else "",
                    "n_flags": rr["n_flags"],
                    "agents_fired": rr["agents_fired"],
                    "bias_types_detected": rr["bias_types_detected"],
                    "severities": rr["severities"],
                    "confidences": rr["confidences"],
                    "correct": int(correct),
                    "reasoning": rr["reasoning"],
                    "api_errors": rr["api_errors"],
                    "ok": int(rr["ok"]),
                    "error": rr.get("error", ""),
                })
            # Crash-safe progress dump every model
            progress_json.write_text(json.dumps({
                "suite": args.suite,
                "analysis_mode": spec["analysis_mode"],
                "completed_model": m_idx, "completed_sentence_in_model": s_idx,
                "elapsed_s": round(time.time() - started, 1),
            }, indent=2))

    # ── Build aggregated summary CSV ─────────────────────────────────────
    summary_cols = [
        "suite", "analysis_mode",
        "sentence_id", "sentence", "expected_label_human", "expected_bias_type",
        "text_type",
        "company", "model",
        "mean_score", "sd_score",
        "agents_consistency",     # pipe-separated set of agents fired across runs
        "bias_types_consistency", # pipe-separated set of bias_types detected across runs
        "correct_count",
        "n_flags_mean",
        "representative_reasoning",
    ]
    summary_csv.write_text(",".join(summary_cols) + "\n")
    grouped: dict[tuple, list[dict]] = {}
    for r in per_run_rows:
        grouped.setdefault((r["sentence_id"], r["model"]), []).append(r)
    with summary_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols, extrasaction="ignore")
        for (sid, mdl), runs in grouped.items():
            ok_scores = [r["score"] for r in runs if r["ok"] and r["score"] != ""]
            mean = round(statistics.mean(ok_scores), 2) if ok_scores else None
            sd = round(statistics.stdev(ok_scores), 2) if len(ok_scores) >= 2 else 0.0
            agents_seen: set[str] = set()
            biases_seen: set[str] = set()
            n_flags_list: list[int] = []
            for r in runs:
                for a in r["agents_fired"].split("|"):
                    if a: agents_seen.add(a)
                for b in r["bias_types_detected"].split("|"):
                    if b: biases_seen.add(b)
                n_flags_list.append(r["n_flags"])
            rep = next((r["reasoning"] for r in runs if r["reasoning"]), "")
            w.writerow({
                "suite": args.suite,
                "analysis_mode": spec["analysis_mode"],
                "sentence_id": sid,
                "sentence": runs[0]["sentence"],
                "expected_label_human": runs[0]["expected_label_human"],
                "expected_bias_type": runs[0]["expected_bias_type"],
                "text_type": runs[0]["text_type"],
                "company": runs[0]["company"],
                "model": mdl,
                "mean_score": mean if mean is not None else "",
                "sd_score": sd,
                "agents_consistency": "|".join(sorted(agents_seen)),
                "bias_types_consistency": "|".join(sorted(biases_seen)),
                "correct_count": sum(r["correct"] for r in runs),
                "n_flags_mean": round(statistics.mean(n_flags_list), 2) if n_flags_list else 0,
                "representative_reasoning": rep,
            })

    print(f"\nDone. Total elapsed: {time.time()-started:.0f}s")
    print(f"Per-run CSV:  {per_run_csv}")
    print(f"Summary CSV:  {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
