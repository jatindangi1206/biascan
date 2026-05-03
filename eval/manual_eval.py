#!/usr/bin/env python3
"""
Manual Evaluation Benchmark — runs the same synthesis text through multiple
models, collects pipeline outputs, and supports human annotation scoring.

Usage:
    # Run benchmark across all configured models
    python -m eval.manual_eval --input-file eval/sample_texts/ms_microbiome.txt

    # Run specific models only
    python -m eval.manual_eval --providers groq,google,mock --samples 1

    # Score existing results (interactive)
    python -m eval.manual_eval --score-file eval/results/manual_benchmark_*.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from backend.models.llm_client import LLMClient
from backend.orchestrator.pipeline import Pipeline
from eval.eval_config import (
    BENCHMARK_MODELS, MANUAL_SCORING_RUBRIC, EVAL_PIPELINE_SETTINGS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("manual_eval")


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ModelRunResult:
    """Result from running one model on the benchmark text."""
    provider: str
    model: str
    label: str                              # human-readable chart label
    elapsed_sec: float = 0.0
    iterations_count: int = 0
    converged: bool = False
    no_bias_detected: bool = False
    final_composite: float = 0.0
    final_sub_scores: dict = field(default_factory=dict)
    convergence_scores: list[float] = field(default_factory=list)  # composite per iteration
    findings_by_agent: dict = field(default_factory=dict)  # {argus: [...], libra: [...], lens: [...]}
    total_findings: int = 0
    total_tokens: int = 0
    token_breakdown: dict = field(default_factory=dict)
    quill_edits_count: int = 0
    vigil_verdicts: list[str] = field(default_factory=list)
    worst_finding: dict = field(default_factory=dict)
    revised_text_preview: str = ""
    error: str = ""

    # Human scores (filled in later via --score-file)
    human_scores: dict = field(default_factory=dict)  # rubric_key -> 1-5 score
    human_notes: str = ""


@dataclass
class BenchmarkReport:
    """Full benchmark report comparing multiple models."""
    input_text_preview: str = ""
    input_text_length: int = 0
    timestamp: str = ""
    total_elapsed_sec: float = 0.0
    models_run: list[ModelRunResult] = field(default_factory=list)
    scoring_rubric: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "input_text_preview": self.input_text_preview,
            "input_text_length": self.input_text_length,
            "timestamp": self.timestamp,
            "total_elapsed_sec": round(self.total_elapsed_sec, 1),
            "models_run": [asdict(m) for m in self.models_run],
            "scoring_rubric": self.scoring_rubric,
        }


# ── Runner ──────────────────────────────────────────────────────────────────

async def run_single_model(
    provider: str,
    model: str,
    label: str,
    text: str,
    api_key: Optional[str] = None,
) -> ModelRunResult:
    """Run the full pipeline for one model and collect results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {label} ({provider}/{model})")

    result = ModelRunResult(provider=provider, model=model, label=label)
    start = time.time()

    try:
        llm = LLMClient(provider=provider, model=model, api_key=api_key)
        pipeline = Pipeline(llm)

        pipe_result = await pipeline.run(
            raw_text=text,
            max_iterations=EVAL_PIPELINE_SETTINGS["max_iterations"],
            patience=EVAL_PIPELINE_SETTINGS["patience"],
            threshold=EVAL_PIPELINE_SETTINGS["threshold"],
        )

        result.elapsed_sec = time.time() - start
        result.iterations_count = len(pipe_result.iterations)
        result.converged = pipe_result.converged
        result.no_bias_detected = pipe_result.no_bias_detected
        result.final_composite = pipe_result.final_composite
        result.final_sub_scores = pipe_result.final_sub_scores

        # Per-iteration convergence curve
        result.convergence_scores = [it.composite_score for it in pipe_result.iterations]

        # Collect findings by agent (from iteration 1 — the initial analysis)
        if pipe_result.iterations:
            first_it = pipe_result.iterations[0]
            result.findings_by_agent = {
                "argus": [
                    {
                        "id": f.id, "bias_type": f.bias_type, "bias_name": f.bias_name,
                        "severity": f.severity.value, "passage": f.passage[:200],
                        "explanation": f.explanation,
                    }
                    for f in first_it.argus_output.findings
                ],
                "libra": [
                    {
                        "id": f.id, "bias_type": f.bias_type, "bias_name": f.bias_name,
                        "severity": f.severity.value, "passage": f.passage[:200],
                        "explanation": f.explanation,
                    }
                    for f in first_it.libra_output.findings
                ],
                "lens": [
                    {
                        "id": f.id, "bias_type": f.bias_type, "bias_name": f.bias_name,
                        "severity": f.severity.value, "passage": f.passage[:200],
                        "explanation": f.explanation,
                    }
                    for f in first_it.lens_output.findings
                ],
            }

        # Total findings across all iterations
        result.total_findings = sum(
            len(it.argus_output.findings) + len(it.libra_output.findings) + len(it.lens_output.findings)
            for it in pipe_result.iterations
        )

        # Token usage
        for it in pipe_result.iterations:
            if it.token_usage:
                for agent_key in ("argus", "libra", "lens", "quill"):
                    agent_tok = it.token_usage.get(agent_key, {})
                    result.token_breakdown[agent_key] = result.token_breakdown.get(agent_key, 0) + agent_tok.get("total_tokens", 0)
                total_tok = it.token_usage.get("total", {})
                result.total_tokens += total_tok.get("total_tokens", 0)

        # QUILL and VIGIL
        result.quill_edits_count = sum(len(it.quill_edits) for it in pipe_result.iterations)
        result.vigil_verdicts = [it.vigil_result.overall.value for it in pipe_result.iterations]

        # Worst finding
        if pipe_result.worst_finding:
            wf = pipe_result.worst_finding
            result.worst_finding = {
                "bias_type": wf.bias_type,
                "bias_name": wf.bias_name,
                "severity": wf.severity.value,
                "passage": wf.passage[:200],
            }

        # Revised text preview
        if pipe_result.final_text:
            result.revised_text_preview = pipe_result.final_text[:500]

        logger.info(f"  Done: composite={result.final_composite:.2f}, "
                     f"findings={result.total_findings}, "
                     f"tokens={result.total_tokens}, "
                     f"{result.elapsed_sec:.1f}s")

    except Exception as e:
        result.elapsed_sec = time.time() - start
        result.error = str(e)
        logger.error(f"  FAILED: {e}")

    return result


async def run_benchmark(
    text: str,
    providers: Optional[list[str]] = None,
    api_keys: Optional[dict[str, str]] = None,
) -> BenchmarkReport:
    """Run the benchmark across all (or selected) models."""
    import datetime

    report = BenchmarkReport(
        input_text_preview=text[:500],
        input_text_length=len(text),
        timestamp=datetime.datetime.now().isoformat(),
        scoring_rubric=MANUAL_SCORING_RUBRIC,
    )

    start = time.time()

    # Filter models by provider if specified
    models_to_run = BENCHMARK_MODELS
    if providers:
        models_to_run = [m for m in BENCHMARK_MODELS if m[0] in providers]

    if not models_to_run:
        logger.error("No models to run! Check --providers argument.")
        return report

    logger.info(f"Running benchmark on {len(models_to_run)} models...")

    # Run models sequentially (rate limits + cleaner logging)
    for provider, model, label in models_to_run:
        api_key = (api_keys or {}).get(provider)
        try:
            result = await run_single_model(provider, model, label, text, api_key)
            report.models_run.append(result)
        except Exception as e:
            logger.error(f"Skipping {label}: {e}")
            report.models_run.append(ModelRunResult(
                provider=provider, model=model, label=label, error=str(e)
            ))

    report.total_elapsed_sec = time.time() - start

    # Save results
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    result_path = results_dir / f"manual_benchmark_{int(time.time())}.json"
    with open(result_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARK COMPLETE in {report.total_elapsed_sec:.1f}s")
    logger.info(f"Results saved to: {result_path}")

    # Summary table
    print(f"\n{'Model':<30} {'Composite':>10} {'Findings':>10} {'Tokens':>10} {'Time':>8} {'Status'}")
    print(f"{'-'*78}")
    for m in report.models_run:
        status = "ERROR" if m.error else ("OK" if not m.no_bias_detected else "CLEAN")
        print(f"{m.label:<30} {m.final_composite:>10.2f} {m.total_findings:>10} "
              f"{m.total_tokens:>10} {m.elapsed_sec:>7.1f}s {status}")
    print()

    return report


# ── Human Scoring ───────────────────────────────────────────────────────────

def interactive_scoring(result_file: str):
    """Load a benchmark result and interactively score each model's output."""
    path = Path(result_file)
    if not path.exists():
        print(f"File not found: {result_file}")
        return

    with open(path) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print("MANUAL SCORING SESSION")
    print(f"Scoring rubric: {json.dumps(data.get('scoring_rubric', {}), indent=2)}")
    print(f"{'='*60}\n")

    for i, model in enumerate(data["models_run"]):
        if model.get("error"):
            print(f"\n[{i+1}] {model['label']} — SKIPPED (error)")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}] {model['label']}")
        print(f"  Composite: {model['final_composite']:.2f}")
        print(f"  Findings: {model['total_findings']}")
        print(f"  Sub-scores: {json.dumps(model.get('final_sub_scores', {}), indent=4)}")

        # Show findings
        for agent, findings in model.get("findings_by_agent", {}).items():
            if findings:
                print(f"\n  --- {agent.upper()} findings ---")
                for f in findings[:5]:
                    print(f"    [{f['severity'].upper()}] {f['bias_type']} — {f['bias_name']}")
                    print(f"      {f['explanation'][:120]}...")

        # Collect scores
        scores = {}
        for rubric_key, rubric_info in data.get("scoring_rubric", {}).items():
            while True:
                try:
                    score = input(f"\n  {rubric_key} ({rubric_info['scale']}): ")
                    score = int(score)
                    if 1 <= score <= 5:
                        scores[rubric_key] = score
                        break
                    print("  Please enter 1-5")
                except (ValueError, EOFError):
                    print("  Please enter 1-5")

        notes = input("  Notes (optional): ")

        model["human_scores"] = scores
        model["human_notes"] = notes

    # Save updated results
    scored_path = path.with_stem(path.stem + "_scored")
    with open(scored_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n  Scored results saved to: {scored_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

# Default test text (MS microbiome systematic review excerpt)
DEFAULT_TEST_TEXT = """Study Selection and Characteristics
The literature search identified 247 records after removal of duplicates. Title and abstract screening excluded 189 records primarily due to irrelevance to multiple sclerosis or absence of microbiome-related outcomes. Fifty-eight full-text articles were assessed for eligibility, of which 46 were excluded because they lacked a healthy comparator, focused on non-human models, or did not report original microbiome data. Twelve observational studies met all inclusion criteria and were included in the final synthesis [1]-[12].

All twelve studies reported significant differences in gut microbiome composition between individuals with MS and healthy controls [1]-[12], although the specific taxa implicated varied. Alpha diversity was reduced in MS in six studies [1], [2], [3], [7], [9], [10], unchanged in four studies [4], [6], [8], [11], and increased in two studies [5], [12]. When pooled descriptively, the direction of effect favored reduced microbial richness in MS, particularly in relapsing-remitting and treatment-naive cohorts [1]-[3], [7], [9].

Consistent patterns included relative depletion of Firmicutes and enrichment of Bacteroidetes and Verrucomicrobia in MS [1], [4], [5], [7], [8], though effect sizes were modest and heterogeneous. Directionally concordant effects were observed in nine of twelve studies for Akkermansia muciniphila, with odds ratios for enrichment in MS ranging from 1.6 to 3.4.

MS was consistently associated with depletion of short-chain fatty acid-producing taxa and reduced abundance of genes involved in butyrate synthesis [6], [11], [12]. Serum metabolomic analyses revealed reduced concentrations of tryptophan-derived metabolites in MS, with large effect sizes (Cohen's d > 0.8) and narrow confidence intervals [6], [10], [12].

Evidence for phenotype-specific signatures was mixed. Three studies found inverse correlations between abundance of butyrate-producing bacteria and disability scores, with Spearman correlation coefficients ranging from -0.22 to -0.40 [3], [11], [12]. Treatment effects were inconsistently reported; however, two studies demonstrated that microbiome differences persisted after adjustment for disease-modifying therapies [1], [6], suggesting that dysbiosis is not solely treatment-driven.

Formal meta-analysis was limited by heterogeneity in sequencing platforms, taxonomic resolution, and outcome reporting. Random-effects synthesis demonstrated substantial heterogeneity, with I-squared values exceeding 70% for alpha-diversity measures. Certainty of evidence was rated as moderate for gut microbiome alterations in MS, low to moderate for specific taxonomic signatures, and moderate for functional disruptions involving short-chain fatty acids and tryptophan metabolism."""


def main():
    parser = argparse.ArgumentParser(description="BiasScan Manual Evaluation Benchmark")
    parser.add_argument("--input-file", default=None, help="Path to synthesis text file")
    parser.add_argument("--providers", default=None, help="Comma-separated providers (e.g. groq,mock)")
    parser.add_argument("--score-file", default=None, help="Score an existing result file interactively")
    args = parser.parse_args()

    # Score existing results
    if args.score_file:
        interactive_scoring(args.score_file)
        return

    # Load input text
    if args.input_file:
        text = Path(args.input_file).read_text()
    else:
        text = DEFAULT_TEST_TEXT
        logger.info("Using default MS microbiome systematic review text")

    # Parse providers
    providers = args.providers.split(",") if args.providers else None

    # Run benchmark
    report = asyncio.run(run_benchmark(text, providers=providers))


if __name__ == "__main__":
    main()
