"""
BiasScan Agent-Wise Benchmark Pipeline.

Flow:
  1. Select one AGENT (argus, libra, lens)
  2. Select one MODEL (gemini, claude, gpt, groq, etc.)
  3. Run that agent on ALL datasets mapped to it
  4. Compute per-dataset metrics (accuracy, precision, recall, FP, FN)
  5. Collate into one score for that model+agent combo
  6. Repeat with different models → agent scoreboard
  7. Repeat for each agent → combined leaderboard

Usage:
  # Generate samples first (only needs to run once)
  python -m eval.benchmark generate --max-samples 20

  # Run one agent with one model
  python -m eval.benchmark run --agent argus --provider groq --model llama-3.3-70b-versatile

  # Run one agent across all configured models
  python -m eval.benchmark sweep --agent argus

  # Run ALL agents across ALL models (full benchmark)
  python -m eval.benchmark full --max-samples 20

  # Generate leaderboard from saved results
  python -m eval.benchmark leaderboard
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval.benchmark")

EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EVAL_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration: Agent → Dataset mapping + Model registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Which datasets belong to which agent
AGENT_DATASETS = {
    "argus": {
        "B01": ["B01_scifact", "B01_ibm_claim"],
        "B03": ["B03_sci_exaggeration", "B03_bioscope", "B03_conll2010"],
        "B04": ["B04_pubhealth", "B04_claimbuster", "B04_sci_exag_scope"],
        "B05": ["B05_media_frames", "B05_mbic", "B05_semeval2023"],
        "B08": ["B08_corr2cause"],
    },
    "libra": {
        "B03": ["LIBRA_bioscope", "LIBRA_sci_exag"],
    },
    "lens": {
        "B01": ["LENS_scicite"],
    },
}

# Models to benchmark (provider, model_id, display_name)
BENCHMARK_MODELS = [
    ("groq", "llama-3.3-70b-versatile", "Llama-3.3-70B"),
    ("groq", "gemma2-9b-it", "Gemma2-9B"),
    ("gemini", "gemini-2.5-flash", "Gemini-2.5-Flash"),
    ("anthropic", "claude-sonnet-4-20250514", "Claude Sonnet 4"),
    ("anthropic", "claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    ("openai", "gpt-4o", "GPT-4o"),
    ("openai", "gpt-4o-mini", "GPT-4o-mini"),
    ("together", "deepseek-ai/DeepSeek-R1", "DeepSeek-R1"),
]

# Bias detection threshold: composite score above this → "biased" prediction
BIAS_THRESHOLD = 0.3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class DatasetScore:
    """Score for one dataset within an agent evaluation."""
    dataset_id: str
    dataset_name: str
    bias_type: str
    total: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    mean_composite_biased: float = 0.0
    mean_composite_control: float = 0.0
    errors: int = 0

    @property
    def accuracy(self) -> float:
        t = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / t if t else 0.0

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def separation(self) -> float:
        """How well the model separates biased from control scores."""
        return self.mean_composite_biased - self.mean_composite_control


@dataclass
class AgentModelScore:
    """Collated score for one model on one agent (across all agent's datasets)."""
    agent: str
    model_name: str
    provider: str
    model_id: str
    dataset_scores: list[DatasetScore] = field(default_factory=list)
    # Collated metrics (computed from dataset_scores)
    collated_accuracy: float = 0.0
    collated_precision: float = 0.0
    collated_recall: float = 0.0
    collated_f1: float = 0.0
    collated_separation: float = 0.0
    total_samples: int = 0
    total_errors: int = 0
    mean_latency: float = 0.0
    timestamp: str = ""

    def compute_collated(self):
        """Compute collated metrics from per-dataset scores."""
        tp = sum(d.true_positives for d in self.dataset_scores)
        tn = sum(d.true_negatives for d in self.dataset_scores)
        fp = sum(d.false_positives for d in self.dataset_scores)
        fn = sum(d.false_negatives for d in self.dataset_scores)
        total = tp + tn + fp + fn

        self.collated_accuracy = (tp + tn) / total if total else 0.0
        self.collated_precision = tp / (tp + fp) if (tp + fp) else 0.0
        self.collated_recall = tp / (tp + fn) if (tp + fn) else 0.0
        p, r = self.collated_precision, self.collated_recall
        self.collated_f1 = 2 * p * r / (p + r) if (p + r) else 0.0

        seps = [d.separation for d in self.dataset_scores if d.total > 0]
        self.collated_separation = sum(seps) / len(seps) if seps else 0.0
        self.total_samples = sum(d.total for d in self.dataset_scores)
        self.total_errors = sum(d.errors for d in self.dataset_scores)


@dataclass
class AgentScoreboard:
    """Scoreboard for one agent across all models."""
    agent: str
    agent_display: str
    bias_types: list[str]
    model_scores: list[AgentModelScore] = field(default_factory=list)

    def ranked(self) -> list[AgentModelScore]:
        """Models ranked by collated F1."""
        return sorted(self.model_scores, key=lambda m: m.collated_f1, reverse=True)


@dataclass
class CombinedLeaderboard:
    """Final combined leaderboard across all agents."""
    agent_scoreboards: dict[str, AgentScoreboard] = field(default_factory=dict)
    overall_ranking: list[dict] = field(default_factory=list)

    def compute_overall(self):
        """Compute overall model ranking by averaging F1 across agents."""
        model_totals: dict[str, list[float]] = {}
        for board in self.agent_scoreboards.values():
            for ms in board.model_scores:
                model_totals.setdefault(ms.model_name, []).append(ms.collated_f1)

        self.overall_ranking = sorted(
            [{"model": k, "mean_f1": sum(v)/len(v), "agents_evaluated": len(v)}
             for k, v in model_totals.items()],
            key=lambda x: x["mean_f1"],
            reverse=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: Generate samples (same as before)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_samples(max_samples: int = 20, seed: int = 42) -> dict:
    """Generate eval samples from all registered transforms."""
    from eval.transforms import get_all_specs
    from dataclasses import asdict

    specs = get_all_specs()
    logger.info(f"Generating samples from {len(specs)} transforms (max {max_samples} per dataset)...")

    all_results = {}
    for spec in specs:
        try:
            samples = spec.transform_fn(max_samples=max_samples, seed=seed)
            biased = [s for s in samples if s.bias_present]
            control = [s for s in samples if not s.bias_present]
            all_results[spec.dataset_id] = {
                "spec": {
                    "dataset_id": spec.dataset_id,
                    "bias_type": spec.bias_type,
                    "dataset_name": spec.dataset_name,
                    "agent": spec.agent,
                    "how_to_use": spec.how_to_use,
                },
                "biased": [asdict(s) for s in biased],
                "control": [asdict(s) for s in control],
            }
            logger.info(f"  ✓ {spec.dataset_id}: {len(biased)} biased + {len(control)} control")
        except Exception as e:
            logger.error(f"  ✗ {spec.dataset_id}: {e}")

    output_path = OUTPUT_DIR / "samples.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nSamples saved: {output_path}")
    logger.info(f"Total datasets: {len(all_results)}")
    return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: Run one agent × one model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_agent_model(
    agent: str,
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    max_samples_per_dataset: int = 10,
) -> AgentModelScore:
    """
    Run one agent across all its datasets with one model.
    Returns collated AgentModelScore.

    Uses the actual BiasScan backend: Orchestrator.analyze() with ProviderConfig.
    The orchestrator runs ALL agents but we only look at the score from our target agent.
    """
    # Add backend to path
    backend_path = str(Path(__file__).resolve().parent.parent / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from app.agents.orchestrator import Orchestrator
    from app.providers.base import ProviderConfig as PC

    # Map agent name to backend agent names
    AGENT_NAME_MAP = {"argus": "ARGUS", "libra": "LIBRA", "lens": "LENS"}
    backend_agent_name = AGENT_NAME_MAP.get(agent, agent.upper())

    # Load samples
    samples_path = OUTPUT_DIR / "samples.json"
    if not samples_path.exists():
        raise FileNotFoundError("No samples.json. Run 'generate' first.")

    with open(samples_path) as f:
        all_samples = json.load(f)

    # Get datasets for this agent
    agent_ds = AGENT_DATASETS.get(agent, {})
    dataset_ids = []
    for bias_type, ds_list in agent_ds.items():
        dataset_ids.extend(ds_list)

    if not dataset_ids:
        raise ValueError(f"No datasets configured for agent '{agent}'")

    # Resolve model name — use provider defaults if not specified
    resolved_model = model or _default_model(provider)
    model_display = resolved_model or "default"

    logger.info(f"\n{'='*70}")
    logger.info(f"AGENT: {agent.upper()} | MODEL: {provider}/{model_display}")
    logger.info(f"Datasets: {len(dataset_ids)}")
    logger.info(f"{'='*70}")

    # Build orchestrator and provider config
    orchestrator = Orchestrator()
    provider_config = PC(
        provider=provider,
        model=resolved_model,
        api_key=api_key or os.environ.get(f"{provider.upper()}_API_KEY", ""),
    )

    result = AgentModelScore(
        agent=agent,
        model_name=f"{provider}/{model_display}",
        provider=provider,
        model_id=resolved_model or "default",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    latencies = []

    for did in dataset_ids:
        if did not in all_samples:
            logger.warning(f"  Dataset {did} not in samples — skipping")
            continue

        data = all_samples[did]
        spec = data["spec"]
        biased_samples = data["biased"][:max_samples_per_dataset // 2]
        control_samples = data["control"][:max_samples_per_dataset // 2]
        eval_samples = biased_samples + control_samples

        if not eval_samples:
            continue

        logger.info(f"\n  Dataset: {spec['dataset_name']} ({did})")
        logger.info(f"  Bias type: {spec['bias_type']} | Samples: {len(eval_samples)}")

        ds_score = DatasetScore(
            dataset_id=did,
            dataset_name=spec["dataset_name"],
            bias_type=spec["bias_type"],
            total=len(eval_samples),
        )

        composite_biased = []
        composite_control = []

        for i, sample in enumerate(eval_samples):
            start = time.time()
            try:
                response = await orchestrator.analyze(
                    text=sample["input_text"],
                    references=None,
                    mode="lite",
                    provider_config=provider_config,
                    agents=[backend_agent_name],
                )
                elapsed = time.time() - start
                latencies.append(elapsed)

                # Use overall_bias_score as the detection signal
                score = response.overall_bias_score
                predicted_biased = score > BIAS_THRESHOLD

                if sample["bias_present"]:
                    composite_biased.append(score)
                    if predicted_biased:
                        ds_score.true_positives += 1
                    else:
                        ds_score.false_negatives += 1
                else:
                    composite_control.append(score)
                    if not predicted_biased:
                        ds_score.true_negatives += 1
                    else:
                        ds_score.false_positives += 1

                status = "✓" if predicted_biased == sample["bias_present"] else "✗"
                logger.info(
                    f"    [{i+1}/{len(eval_samples)}] {status} "
                    f"score={score:.2f} "
                    f"pred={'B' if predicted_biased else 'C'} "
                    f"exp={'B' if sample['bias_present'] else 'C'} "
                    f"ann={len(response.annotations)} "
                    f"[{elapsed:.1f}s]"
                )

            except Exception as e:
                elapsed = time.time() - start
                ds_score.errors += 1
                logger.error(f"    [{i+1}/{len(eval_samples)}] ERROR: {e} [{elapsed:.1f}s]")

        ds_score.mean_composite_biased = (
            sum(composite_biased) / len(composite_biased) if composite_biased else 0.0
        )
        ds_score.mean_composite_control = (
            sum(composite_control) / len(composite_control) if composite_control else 0.0
        )

        result.dataset_scores.append(ds_score)

        logger.info(
            f"  → Acc={ds_score.accuracy:.3f} P={ds_score.precision:.3f} "
            f"R={ds_score.recall:.3f} F1={ds_score.f1:.3f} "
            f"Sep={ds_score.separation:.3f}"
        )

    # Compute collated score
    result.mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    result.compute_collated()

    logger.info(f"\n  {'─'*50}")
    logger.info(f"  COLLATED SCORE for {agent.upper()} × {provider}/{model_display}:")
    logger.info(f"    Accuracy={result.collated_accuracy:.3f}")
    logger.info(f"    Precision={result.collated_precision:.3f}")
    logger.info(f"    Recall={result.collated_recall:.3f}")
    logger.info(f"    F1={result.collated_f1:.3f}")
    logger.info(f"    Separation={result.collated_separation:.3f}")
    logger.info(f"    Samples={result.total_samples} Errors={result.total_errors}")
    logger.info(f"    Mean latency={result.mean_latency:.1f}s")

    # Save individual result
    result_path = RESULTS_DIR / f"{agent}_{provider}_{resolved_model or 'default'}_{int(time.time())}.json"
    with open(result_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    logger.info(f"  Saved: {result_path}")

    return result


def _default_model(provider: str) -> str:
    """Get default model for a provider."""
    defaults = {
        "groq": "llama-3.3-70b-versatile",
        "gemini": "gemini-2.5-flash",
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "ollama": "qwen2.5:7b",
        "nvidia": "meta/llama-3.1-70b-instruct",
    }
    return defaults.get(provider, "")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: Sweep one agent across all models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AGENT_DISPLAY_NAMES = {
    "argus": "ARGUS — Bias Detection",
    "libra": "LIBRA — Scope & Hedging",
    "lens": "LENS — Discourse Analysis",
}

AGENT_BIAS_TYPES = {
    "argus": ["B01 Confirmation", "B03 Certainty", "B04 Overgen.", "B05 Framing", "B08 Causal"],
    "libra": ["B03 Hedging/Scope"],
    "lens": ["B01 Citation Intent"],
}


async def sweep_agent(
    agent: str,
    models: list[tuple[str, str, str]] | None = None,
    max_samples_per_dataset: int = 10,
) -> AgentScoreboard:
    """Run all models for one agent and build scoreboard."""
    models = models or BENCHMARK_MODELS

    board = AgentScoreboard(
        agent=agent,
        agent_display=AGENT_DISPLAY_NAMES.get(agent, agent.upper()),
        bias_types=AGENT_BIAS_TYPES.get(agent, []),
    )

    for provider, model_id, display_name in models:
        logger.info(f"\n{'━'*70}")
        logger.info(f"SWEEP: {agent.upper()} × {display_name} ({provider}/{model_id})")
        logger.info(f"{'━'*70}")

        try:
            score = await run_agent_model(
                agent=agent,
                provider=provider,
                model=model_id,
                max_samples_per_dataset=max_samples_per_dataset,
            )
            score.model_name = display_name  # Use friendly name
            board.model_scores.append(score)
        except Exception as e:
            logger.error(f"FAILED: {display_name} — {e}")

    # Print agent scoreboard
    _print_agent_scoreboard(board)
    return board


def _print_agent_scoreboard(board: AgentScoreboard):
    """Print formatted agent scoreboard."""
    ranked = board.ranked()
    print(f"\n{'═'*80}")
    print(f"  {board.agent_display} — SCOREBOARD")
    print(f"  Bias Types: {', '.join(board.bias_types)}")
    print(f"{'═'*80}")
    print(f"  {'Rank':<5} {'Model':<25} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Acc':>7} {'Sep':>7}")
    print(f"  {'─'*73}")

    for i, ms in enumerate(ranked):
        print(
            f"  {i+1:<5} {ms.model_name[:24]:<25} "
            f"{ms.collated_f1:>7.3f} {ms.collated_precision:>7.3f} "
            f"{ms.collated_recall:>7.3f} {ms.collated_accuracy:>7.3f} "
            f"{ms.collated_separation:>7.3f}"
        )

    print(f"{'═'*80}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: Full benchmark (all agents × all models)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def full_benchmark(
    max_samples: int = 20,
    models: list[tuple[str, str, str]] | None = None,
    seed: int = 42,
) -> CombinedLeaderboard:
    """Run full benchmark: all agents × all models."""
    # Generate samples first
    generate_samples(max_samples=max_samples, seed=seed)

    leaderboard = CombinedLeaderboard()

    for agent in AGENT_DATASETS.keys():
        board = await sweep_agent(
            agent=agent,
            models=models,
            max_samples_per_dataset=max_samples,
        )
        leaderboard.agent_scoreboards[agent] = board

    leaderboard.compute_overall()

    # Print combined leaderboard
    print(f"\n{'█'*80}")
    print(f"  BIASSCAN COMBINED LEADERBOARD")
    print(f"{'█'*80}")
    print(f"  {'Rank':<5} {'Model':<25} {'Mean F1':>8} {'Agents':>8}")
    print(f"  {'─'*50}")
    for i, entry in enumerate(leaderboard.overall_ranking):
        print(f"  {i+1:<5} {entry['model'][:24]:<25} {entry['mean_f1']:>8.3f} {entry['agents_evaluated']:>8}")
    print(f"{'█'*80}\n")

    # Save combined results
    combined_path = OUTPUT_DIR / f"combined_leaderboard_{int(time.time())}.json"
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_ranking": leaderboard.overall_ranking,
        "agent_scoreboards": {},
    }
    for agent, board in leaderboard.agent_scoreboards.items():
        save_data["agent_scoreboards"][agent] = {
            "agent_display": board.agent_display,
            "bias_types": board.bias_types,
            "model_scores": [asdict(ms) for ms in board.model_scores],
        }
    with open(combined_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    logger.info(f"Combined leaderboard saved: {combined_path}")
    return leaderboard


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: Build leaderboard from saved results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_leaderboard_from_results() -> CombinedLeaderboard:
    """Reconstruct leaderboard from saved result JSON files."""
    leaderboard = CombinedLeaderboard()

    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if not result_files:
        logger.error("No result files found in output/results/")
        return leaderboard

    logger.info(f"Loading {len(result_files)} result files...")

    # Group by agent
    agent_results: dict[str, list[AgentModelScore]] = {}
    for f in result_files:
        with open(f) as fh:
            data = json.load(fh)

        agent = data["agent"]
        ds_scores = []
        for ds in data.get("dataset_scores", []):
            ds_scores.append(DatasetScore(**{k: v for k, v in ds.items()
                                             if k in DatasetScore.__dataclass_fields__}))

        ms = AgentModelScore(
            agent=agent,
            model_name=data["model_name"],
            provider=data["provider"],
            model_id=data["model_id"],
            dataset_scores=ds_scores,
            timestamp=data.get("timestamp", ""),
        )
        ms.compute_collated()
        ms.mean_latency = data.get("mean_latency", 0.0)

        agent_results.setdefault(agent, []).append(ms)

    for agent, scores in agent_results.items():
        board = AgentScoreboard(
            agent=agent,
            agent_display=AGENT_DISPLAY_NAMES.get(agent, agent.upper()),
            bias_types=AGENT_BIAS_TYPES.get(agent, []),
            model_scores=scores,
        )
        leaderboard.agent_scoreboards[agent] = board
        _print_agent_scoreboard(board)

    leaderboard.compute_overall()
    return leaderboard


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="BiasScan Agent-Wise Benchmark")
    sub = parser.add_subparsers(dest="command")

    # Generate samples
    gen = sub.add_parser("generate", help="Generate eval samples from transforms")
    gen.add_argument("--max-samples", type=int, default=20)
    gen.add_argument("--seed", type=int, default=42)

    # Run one agent × one model
    run = sub.add_parser("run", help="Run one agent with one model")
    run.add_argument("--agent", required=True, choices=["argus", "libra", "lens"])
    run.add_argument("--provider", required=True)
    run.add_argument("--model", default=None)
    run.add_argument("--api-key", default=None)
    run.add_argument("--max-samples", type=int, default=10)

    # Sweep one agent across all models
    sweep = sub.add_parser("sweep", help="Sweep one agent across all configured models")
    sweep.add_argument("--agent", required=True, choices=["argus", "libra", "lens"])
    sweep.add_argument("--max-samples", type=int, default=10)

    # Full benchmark
    full = sub.add_parser("full", help="Full benchmark: all agents × all models")
    full.add_argument("--max-samples", type=int, default=20)
    full.add_argument("--seed", type=int, default=42)

    # Build leaderboard from existing results
    lb = sub.add_parser("leaderboard", help="Build leaderboard from saved results")

    args = parser.parse_args()

    if args.command == "generate":
        generate_samples(max_samples=args.max_samples, seed=args.seed)

    elif args.command == "run":
        asyncio.run(run_agent_model(
            agent=args.agent,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            max_samples_per_dataset=args.max_samples,
        ))

    elif args.command == "sweep":
        asyncio.run(sweep_agent(
            agent=args.agent,
            max_samples_per_dataset=args.max_samples,
        ))

    elif args.command == "full":
        asyncio.run(full_benchmark(
            max_samples=args.max_samples,
            seed=args.seed,
        ))

    elif args.command == "leaderboard":
        build_leaderboard_from_results()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
