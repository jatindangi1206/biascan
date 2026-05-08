"""
Generate the BiasScan Benchmarks HTML page.

Reads results from eval/output/results/ and produces a standalone HTML page
with per-agent scoreboards and a combined leaderboard.

Usage:
  python -m eval.generate_leaderboard_html
  python -m eval.generate_leaderboard_html --output frontend/public/benchmarks.html
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EVAL_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "results"


def generate_html(output_path: str | None = None) -> str:
    """Generate leaderboard HTML from saved results."""
    from eval.benchmark import build_leaderboard_from_results, AGENT_DISPLAY_NAMES

    leaderboard = build_leaderboard_from_results()

    # Build HTML
    agent_sections = []
    for agent, board in leaderboard.agent_scoreboards.items():
        ranked = board.ranked()
        if not ranked:
            continue

        rows = ""
        for i, ms in enumerate(ranked):
            # Per-dataset breakdown
            ds_details = []
            for ds in ms.dataset_scores:
                ds_details.append(
                    f"{ds.dataset_name}: F1={ds.f1:.3f} P={ds.precision:.3f} R={ds.recall:.3f}"
                )
            tooltip = " | ".join(ds_details)

            badge = ""
            if i == 0:
                badge = '<span class="badge gold">🥇</span>'
            elif i == 1:
                badge = '<span class="badge silver">🥈</span>'
            elif i == 2:
                badge = '<span class="badge bronze">🥉</span>'

            rows += f"""
            <tr title="{tooltip}">
                <td class="rank">{i+1} {badge}</td>
                <td class="model-name">{ms.model_name}</td>
                <td class="metric f1">{ms.collated_f1:.3f}</td>
                <td class="metric">{ms.collated_precision:.3f}</td>
                <td class="metric">{ms.collated_recall:.3f}</td>
                <td class="metric">{ms.collated_accuracy:.3f}</td>
                <td class="metric sep">{ms.collated_separation:.3f}</td>
                <td class="metric latency">{ms.mean_latency:.1f}s</td>
            </tr>"""

        # Dataset coverage info
        datasets_used = set()
        for ms in ranked:
            for ds in ms.dataset_scores:
                datasets_used.add(f"{ds.dataset_name} ({ds.bias_type})")

        datasets_list = ", ".join(sorted(datasets_used))

        agent_sections.append(f"""
        <section class="agent-board" id="{agent}">
            <h2>{board.agent_display}</h2>
            <p class="bias-types">Evaluates: {', '.join(board.bias_types)}</p>
            <p class="datasets-used">Datasets: {datasets_list}</p>
            <table class="scoreboard">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>F1 ↓</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>Accuracy</th>
                        <th>Separation</th>
                        <th>Latency</th>
                    </tr>
                </thead>
                <tbody>{rows}
                </tbody>
            </table>
        </section>""")

    # Combined leaderboard
    overall_rows = ""
    for i, entry in enumerate(leaderboard.overall_ranking):
        badge = ""
        if i == 0:
            badge = '<span class="badge gold">🥇</span>'
        elif i == 1:
            badge = '<span class="badge silver">🥈</span>'
        elif i == 2:
            badge = '<span class="badge bronze">🥉</span>'

        overall_rows += f"""
            <tr>
                <td class="rank">{i+1} {badge}</td>
                <td class="model-name">{entry['model']}</td>
                <td class="metric f1">{entry['mean_f1']:.3f}</td>
                <td class="metric">{entry['agents_evaluated']}</td>
            </tr>"""

    timestamp = time.strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BiasScan Benchmarks — Model Leaderboard</title>
    <style>
        :root {{
            --bg: #0f1117;
            --surface: #1a1d27;
            --border: #2a2d3a;
            --text: #e4e4e7;
            --text-muted: #9ca3af;
            --accent: #6366f1;
            --accent-light: #818cf8;
            --gold: #fbbf24;
            --silver: #9ca3af;
            --bronze: #d97706;
            --green: #22c55e;
            --red: #ef4444;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, var(--surface), #1e2030);
            border: 1px solid var(--border);
            border-radius: 12px;
        }}

        header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--accent-light), #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        header p {{
            color: var(--text-muted);
            font-size: 1.1rem;
        }}

        .nav-links {{
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }}

        .nav-links a {{
            color: var(--accent-light);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            transition: all 0.2s;
        }}

        .nav-links a:hover {{
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }}

        .combined-board {{
            margin-bottom: 3rem;
            padding: 2rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
        }}

        .combined-board h2 {{
            font-size: 1.8rem;
            margin-bottom: 1rem;
            color: var(--accent-light);
        }}

        .agent-board {{
            margin-bottom: 2.5rem;
            padding: 1.5rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
        }}

        .agent-board h2 {{
            font-size: 1.4rem;
            margin-bottom: 0.5rem;
        }}

        .bias-types {{
            color: var(--accent-light);
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }}

        .datasets-used {{
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }}

        table.scoreboard {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }}

        table.scoreboard th {{
            text-align: left;
            padding: 0.75rem 1rem;
            border-bottom: 2px solid var(--border);
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        table.scoreboard td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
        }}

        table.scoreboard tr:hover {{
            background: rgba(99, 102, 241, 0.05);
        }}

        .rank {{ width: 80px; }}
        .model-name {{ font-weight: 600; }}
        .metric {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; text-align: right; }}
        .f1 {{ color: var(--green); font-weight: 700; }}
        .sep {{ color: var(--accent-light); }}
        .latency {{ color: var(--text-muted); }}

        .badge {{
            font-size: 0.8rem;
            vertical-align: middle;
        }}

        .methodology {{
            margin-top: 3rem;
            padding: 2rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
        }}

        .methodology h2 {{
            margin-bottom: 1rem;
            color: var(--accent-light);
        }}

        .methodology h3 {{
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            color: var(--text);
        }}

        .methodology p, .methodology li {{
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .methodology ul {{
            padding-left: 1.5rem;
        }}

        .timestamp {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 2rem;
        }}

        @media (max-width: 768px) {{
            body {{ padding: 1rem; }}
            header h1 {{ font-size: 1.8rem; }}
            table.scoreboard {{ font-size: 0.8rem; }}
            table.scoreboard td, table.scoreboard th {{ padding: 0.5rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>BiasScan Benchmarks</h1>
            <p>Multi-model evaluation of cognitive bias detection in AI-generated systematic review text</p>
            <div class="nav-links">
                <a href="#combined">Combined Leaderboard</a>
                <a href="#argus">ARGUS</a>
                <a href="#libra">LIBRA</a>
                <a href="#lens">LENS</a>
                <a href="#methodology">Methodology</a>
            </div>
        </header>

        <section class="combined-board" id="combined">
            <h2>Combined Leaderboard</h2>
            <p style="color: var(--text-muted); margin-bottom: 1rem;">
                Overall model ranking by mean F1 score across all agents.
            </p>
            <table class="scoreboard">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Mean F1 ↓</th>
                        <th>Agents Evaluated</th>
                    </tr>
                </thead>
                <tbody>{overall_rows}
                </tbody>
            </table>
        </section>

        {"".join(agent_sections)}

        <section class="methodology" id="methodology">
            <h2>Methodology</h2>
            <p>BiasScan evaluates LLM-powered agents that detect cognitive bias in systematic review synthesis text.</p>

            <h3>Agents</h3>
            <ul>
                <li><strong>ARGUS</strong> — Detects 11 cognitive bias types (confirmation, certainty inflation, overgeneralization, framing, causal inference errors)</li>
                <li><strong>LIBRA</strong> — Analyzes hedging, boosting, and scope calibration using Hyland linguistic markers</li>
                <li><strong>LENS</strong> — Evaluates discourse structure, citation intent, and argumentation quality</li>
            </ul>

            <h3>Evaluation Protocol</h3>
            <ul>
                <li>Each agent is evaluated independently on datasets mapped to its bias types</li>
                <li>Datasets include both <strong>biased</strong> (transformed to contain specific bias patterns) and <strong>control</strong> (neutral) samples</li>
                <li>A composite score above 0.3 is classified as "bias detected"</li>
                <li>Metrics: Accuracy, Precision, Recall, F1, and Score Separation (mean biased score − mean control score)</li>
            </ul>

            <h3>Datasets</h3>
            <ul>
                <li><strong>SciFact</strong> (AllenAI) — Scientific claim verification with evidence labels</li>
                <li><strong>IBM Claim Stance</strong> — Pro/Con claims for debate topics</li>
                <li><strong>Scientific Exaggeration</strong> (CopenNLU) — Abstract vs press release overclaiming</li>
                <li><strong>BioScope</strong> — Biomedical hedging annotations</li>
                <li><strong>PubHealth</strong> — Health claim veracity labels</li>
                <li><strong>Corr2Cause</strong> (ICLR 2024) — Correlation vs causation inference</li>
                <li><strong>SciCite</strong> (AllenAI) — Citation intent classification</li>
                <li><strong>Media Frames Corpus</strong> — 15-dimension framing annotations</li>
                <li><strong>MBIC</strong> — Word-level media bias annotations</li>
                <li><strong>SemEval-2023 Task 3</strong> — Persuasion technique detection</li>
            </ul>

            <h3>Scoring</h3>
            <p>
                Per-agent scores are computed by running the model on all datasets for that agent,
                computing per-dataset TP/TN/FP/FN, then collating across datasets for overall P/R/F1.
                The combined leaderboard averages F1 scores across all agents.
            </p>
        </section>

        <p class="timestamp">Last updated: {timestamp}</p>
    </div>
</body>
</html>"""

    # Save
    if output_path is None:
        output_path = str(OUTPUT_DIR / "benchmarks.html")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Leaderboard HTML saved: {output_path}")
    return html


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Output HTML file path")
    args = parser.parse_args()
    generate_html(output_path=args.output)
