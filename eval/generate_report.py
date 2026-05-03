#!/usr/bin/env python3
"""
HTML Report Generator — creates self-contained HTML benchmark reports
with Chart.js charts from eval JSON result files.

Usage:
    # Generate report from automated eval results
    python -m eval.generate_report --auto eval/results/auto_eval_*.json

    # Generate report from manual benchmark results
    python -m eval.generate_report --manual eval/results/manual_benchmark_*.json

    # Generate combined report
    python -m eval.generate_report --auto eval/results/auto_*.json --manual eval/results/manual_*.json
"""

from __future__ import annotations

import argparse
import json
import glob
import sys
from pathlib import Path
from typing import Optional


def _load_json_files(pattern: str) -> list[dict]:
    """Load all JSON files matching a glob pattern."""
    files = sorted(glob.glob(pattern))
    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def generate_html_report(
    auto_results: list[dict] = None,
    manual_results: list[dict] = None,
    output_path: str = "eval/reports/benchmark_report.html",
) -> str:
    """Generate a self-contained HTML report with Chart.js charts."""

    auto_results = auto_results or []
    manual_results = manual_results or []

    # ── Prepare auto eval data ──────────────────────────────────────────
    auto_section = ""
    auto_chart_js = ""

    if auto_results:
        # Collect all dataset metrics across models
        auto_rows = []
        for result in auto_results:
            model = result.get("model_used", "unknown")
            for ds in result.get("datasets", []):
                auto_rows.append({
                    "model": model,
                    "dataset": ds["dataset_name"],
                    "precision": ds.get("precision", 0),
                    "recall": ds.get("recall", 0),
                    "f1": ds.get("f1", 0),
                    "accuracy": ds.get("accuracy", 0),
                    "tp": ds.get("true_positives", 0),
                    "fp": ds.get("false_positives", 0),
                    "tn": ds.get("true_negatives", 0),
                    "fn": ds.get("false_negatives", 0),
                    "tokens": ds.get("total_tokens", 0),
                    "avg_time": round(ds.get("avg_elapsed_sec", 0), 1),
                })

        auto_table_rows = "\n".join(
            f"""<tr>
                <td>{r['model']}</td><td>{r['dataset']}</td>
                <td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td>
                <td><strong>{r['f1']:.3f}</strong></td><td>{r['accuracy']:.3f}</td>
                <td>{r['tp']}</td><td>{r['fp']}</td><td>{r['tn']}</td><td>{r['fn']}</td>
                <td>{r['tokens']:,}</td><td>{r['avg_time']}s</td>
            </tr>"""
            for r in auto_rows
        )

        # Build per-model summary for bar chart
        model_summaries = {}
        for result in auto_results:
            model = result.get("model_used", "unknown")
            summary = result.get("summary", {})
            model_summaries[model] = summary

        auto_chart_labels = json.dumps(list(model_summaries.keys()))
        auto_chart_p = json.dumps([s.get("avg_precision", 0) for s in model_summaries.values()])
        auto_chart_r = json.dumps([s.get("avg_recall", 0) for s in model_summaries.values()])
        auto_chart_f1 = json.dumps([s.get("avg_f1", 0) for s in model_summaries.values()])

        auto_chart_js = f"""
        new Chart(document.getElementById('autoBarChart'), {{
            type: 'bar',
            data: {{
                labels: {auto_chart_labels},
                datasets: [
                    {{ label: 'Precision', data: {auto_chart_p}, backgroundColor: 'rgba(54, 162, 235, 0.7)' }},
                    {{ label: 'Recall', data: {auto_chart_r}, backgroundColor: 'rgba(255, 99, 132, 0.7)' }},
                    {{ label: 'F1', data: {auto_chart_f1}, backgroundColor: 'rgba(75, 192, 192, 0.7)' }},
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Automated Eval: Bias Detection Metrics by Model' }} }},
                scales: {{ y: {{ beginAtZero: true, max: 1.0 }} }}
            }}
        }});
        """

        auto_section = f"""
        <section id="auto-eval">
            <h2>Automated Evaluation Results</h2>
            <p>Bias detection accuracy measured against labeled datasets (SciFact, PubHealth, Corr2Cause, etc.)</p>

            <div class="chart-container">
                <canvas id="autoBarChart"></canvas>
            </div>

            <h3>Detailed Results</h3>
            <table>
                <thead>
                    <tr>
                        <th>Model</th><th>Dataset</th>
                        <th>Precision</th><th>Recall</th><th>F1</th><th>Accuracy</th>
                        <th>TP</th><th>FP</th><th>TN</th><th>FN</th>
                        <th>Tokens</th><th>Avg Time</th>
                    </tr>
                </thead>
                <tbody>
                    {auto_table_rows}
                </tbody>
            </table>
        </section>
        """

    # ── Prepare manual eval data ────────────────────────────────────────
    manual_section = ""
    manual_chart_js = ""

    if manual_results:
        for bench in manual_results:
            models = bench.get("models_run", [])

            # Comparison table
            manual_table_rows = "\n".join(
                f"""<tr class="{'error-row' if m.get('error') else ''}">
                    <td>{m['label']}</td>
                    <td><strong>{m.get('final_composite', 0):.2f}</strong></td>
                    <td>{m.get('total_findings', 0)}</td>
                    <td>{json.dumps(m.get('final_sub_scores', {}))}</td>
                    <td>{m.get('total_tokens', 0):,}</td>
                    <td>{m.get('elapsed_sec', 0):.1f}s</td>
                    <td>{m.get('quill_edits_count', 0)}</td>
                    <td>{', '.join(m.get('vigil_verdicts', []))}</td>
                    <td>{m.get('error', '') or 'OK'}</td>
                </tr>"""
                for m in models
            )

            # Radar chart data (sub-scores for each model)
            radar_labels = set()
            for m in models:
                radar_labels.update(m.get("final_sub_scores", {}).keys())
            radar_labels = sorted(radar_labels)

            radar_datasets = []
            colors = [
                "rgba(255, 99, 132, 0.5)", "rgba(54, 162, 235, 0.5)",
                "rgba(255, 206, 86, 0.5)", "rgba(75, 192, 192, 0.5)",
                "rgba(153, 102, 255, 0.5)", "rgba(255, 159, 64, 0.5)",
                "rgba(199, 199, 199, 0.5)", "rgba(83, 102, 255, 0.5)",
                "rgba(255, 99, 255, 0.5)", "rgba(99, 255, 132, 0.5)",
            ]
            for idx, m in enumerate(models):
                if m.get("error"):
                    continue
                sub = m.get("final_sub_scores", {})
                data = [sub.get(k, 0) for k in radar_labels]
                border_color = colors[idx % len(colors)].replace("0.5", "1.0")
                radar_datasets.append({
                    "label": m["label"],
                    "data": data,
                    "backgroundColor": colors[idx % len(colors)],
                    "borderColor": border_color,
                    "borderWidth": 2,
                })

            # Convergence line chart
            conv_datasets = []
            for idx, m in enumerate(models):
                if m.get("error") or not m.get("convergence_scores"):
                    continue
                border_color = colors[idx % len(colors)].replace("0.5", "1.0")
                conv_datasets.append({
                    "label": m["label"],
                    "data": m["convergence_scores"],
                    "borderColor": border_color,
                    "fill": False,
                    "tension": 0.3,
                })

            max_iters = max((len(m.get("convergence_scores", [])) for m in models), default=1)
            conv_labels = [f"Iter {i+1}" for i in range(max_iters)]

            # Token cost bar chart
            token_labels = [m["label"] for m in models if not m.get("error")]
            token_data = [m.get("total_tokens", 0) for m in models if not m.get("error")]

            # Human scores comparison (if scored)
            human_section = ""
            scored_models = [m for m in models if m.get("human_scores")]
            if scored_models:
                rubric_keys = list(bench.get("scoring_rubric", {}).keys())
                human_rows = "\n".join(
                    f"""<tr>
                        <td>{m['label']}</td>
                        {''.join(f"<td>{m['human_scores'].get(k, '-')}</td>" for k in rubric_keys)}
                        <td>{sum(m['human_scores'].values()) / len(m['human_scores']) if m['human_scores'] else 0:.1f}</td>
                        <td>{m.get('human_notes', '')}</td>
                    </tr>"""
                    for m in scored_models
                )
                human_section = f"""
                <h3>Human Evaluation Scores</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            {''.join(f"<th>{k}</th>" for k in rubric_keys)}
                            <th>Average</th>
                            <th>Notes</th>
                        </tr>
                    </thead>
                    <tbody>{human_rows}</tbody>
                </table>
                """

            manual_chart_js += f"""
            new Chart(document.getElementById('radarChart'), {{
                type: 'radar',
                data: {{
                    labels: {json.dumps(radar_labels)},
                    datasets: {json.dumps(radar_datasets)}
                }},
                options: {{
                    responsive: true,
                    plugins: {{ title: {{ display: true, text: 'Sub-Score Comparison (Radar)' }} }},
                    scales: {{ r: {{ beginAtZero: true, max: 10 }} }}
                }}
            }});

            new Chart(document.getElementById('convergenceChart'), {{
                type: 'line',
                data: {{
                    labels: {json.dumps(conv_labels)},
                    datasets: {json.dumps(conv_datasets)}
                }},
                options: {{
                    responsive: true,
                    plugins: {{ title: {{ display: true, text: 'Convergence Curve (Composite Score per Iteration)' }} }},
                    scales: {{ y: {{ beginAtZero: true }} }}
                }}
            }});

            new Chart(document.getElementById('tokenChart'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(token_labels)},
                    datasets: [{{
                        label: 'Total Tokens',
                        data: {json.dumps(token_data)},
                        backgroundColor: 'rgba(153, 102, 255, 0.7)',
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{ title: {{ display: true, text: 'Token Usage by Model' }} }},
                    scales: {{ y: {{ beginAtZero: true }} }}
                }}
            }});
            """

            manual_section += f"""
            <section id="manual-eval">
                <h2>Manual Benchmark Results</h2>
                <p>Same synthesis text ({bench.get('input_text_length', 0):,} chars) run through {len(models)} models.</p>

                <div class="charts-grid">
                    <div class="chart-container"><canvas id="radarChart"></canvas></div>
                    <div class="chart-container"><canvas id="convergenceChart"></canvas></div>
                    <div class="chart-container"><canvas id="tokenChart"></canvas></div>
                </div>

                <h3>Model Comparison Table</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th><th>Composite</th><th>Findings</th>
                            <th>Sub-Scores</th><th>Tokens</th><th>Time</th>
                            <th>QUILL Edits</th><th>VIGIL</th><th>Status</th>
                        </tr>
                    </thead>
                    <tbody>{manual_table_rows}</tbody>
                </table>

                {human_section}
            </section>
            """

    # ── Assemble HTML ───────────────────────────────────────────────────

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BiasScan Evaluation Report</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
    <style>
        :root {{
            --bg: #0f172a;
            --card: #1e293b;
            --text: #e2e8f0;
            --accent: #38bdf8;
            --border: #334155;
            --green: #4ade80;
            --red: #f87171;
            --yellow: #fbbf24;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
            line-height: 1.6;
        }}
        h1 {{
            font-size: 2rem;
            background: linear-gradient(135deg, var(--accent), #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        h2 {{
            font-size: 1.5rem;
            color: var(--accent);
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }}
        h3 {{ font-size: 1.1rem; color: var(--text); margin: 1.5rem 0 0.75rem; }}
        p {{ color: #94a3b8; margin-bottom: 1rem; }}
        .header {{ margin-bottom: 2rem; }}
        .header .meta {{ color: #64748b; font-size: 0.85rem; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0 2rem;
            font-size: 0.85rem;
        }}
        th, td {{
            padding: 0.6rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: var(--card);
            color: var(--accent);
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        tr:hover {{ background: rgba(56, 189, 248, 0.05); }}
        .error-row {{ opacity: 0.5; }}

        .chart-container {{
            background: var(--card);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid var(--border);
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1rem;
        }}

        .nav {{
            position: sticky;
            top: 0;
            background: var(--bg);
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1.5rem;
            z-index: 10;
        }}
        .nav a {{
            color: var(--accent);
            text-decoration: none;
            margin-right: 1.5rem;
            font-size: 0.9rem;
        }}
        .nav a:hover {{ text-decoration: underline; }}

        strong {{ color: var(--green); }}
        .tag {{
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .tag-pass {{ background: rgba(74, 222, 128, 0.2); color: var(--green); }}
        .tag-fail {{ background: rgba(248, 113, 113, 0.2); color: var(--red); }}
    </style>
</head>
<body>

<div class="header">
    <h1>BiasScan Evaluation Report</h1>
    <p class="meta">Generated by BiasScan eval framework</p>
</div>

<nav class="nav">
    {'<a href="#auto-eval">Automated Eval</a>' if auto_results else ''}
    {'<a href="#manual-eval">Manual Benchmark</a>' if manual_results else ''}
</nav>

{auto_section}
{manual_section}

<script>
{auto_chart_js}
{manual_chart_js}
</script>

</body>
</html>"""

    # Write report
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"Report generated: {out_path}")
    return str(out_path)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate BiasScan HTML Evaluation Report")
    parser.add_argument("--auto", default=None, help="Glob pattern for auto eval JSON files")
    parser.add_argument("--manual", default=None, help="Glob pattern for manual benchmark JSON files")
    parser.add_argument("--output", default="eval/reports/benchmark_report.html", help="Output HTML path")
    args = parser.parse_args()

    auto_results = _load_json_files(args.auto) if args.auto else []
    manual_results = _load_json_files(args.manual) if args.manual else []

    if not auto_results and not manual_results:
        print("No result files provided. Use --auto and/or --manual with glob patterns.")
        sys.exit(1)

    generate_html_report(auto_results, manual_results, args.output)


if __name__ == "__main__":
    main()
