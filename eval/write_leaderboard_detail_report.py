from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = ROOT / "eval" / "output" / "leaderboard.json"
RERUN = ROOT / "eval" / "output" / "leaderboard_rerun.json"
OUT = ROOT / "eval" / "output" / "leaderboard_detailed_report.md"

PAPERS_ORDER = ["Nu-OG", "Nu-OG-GPT", "Nu-Edit", "Nu-bias-injected"]


def fmt_num(value: float) -> str:
    return f"{value:.2f}"


def fmt_ts(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def status(result: dict) -> str:
    if "crashed" in result:
        return "CRASHED"

    saw_valid_model_id_error = False
    papers = result.get("papers", {})
    all_failed = True
    all_invalid = True

    for paper_data in papers.values():
        elapsed = paper_data.get("elapsed_s", 0)
        mean = paper_data.get("mean", 0)
        runs = paper_data.get("runs", [])
        api_fails = [run.get("api_failure") for run in runs if isinstance(run, dict)]

        if api_fails and all(api_fails):
            pass
        else:
            if elapsed and elapsed > 2:
                all_failed = False
            if mean > 0 or elapsed > 2:
                all_invalid = False

        for run in runs:
            if not isinstance(run, dict):
                continue
            for warning in run.get("warnings", [])[:8]:
                if "is not a valid model ID" in warning:
                    saw_valid_model_id_error = True

    if saw_valid_model_id_error:
        return "INVALID_ID"
    if all_invalid:
        return "API_FAIL"
    if all_failed and papers:
        return "API_FAIL"
    return "OK"


def sortkey(result: dict) -> tuple[int, float, str]:
    result_status = status(result)
    if result_status != "OK":
        return (1, 0.0, result["model"])
    injected = result.get("papers", {}).get("Nu-bias-injected", {}).get("mean", 0.0)
    return (0, -injected, result["model"])


def summarize_run(run: dict, index: int) -> list[str]:
    lines = [f"#### Run {index}"]
    lines.append("")
    lines.append(f"- `ok`: `{run.get('ok', False)}`")

    score = run.get("score")
    if score is None:
        lines.append("- `score`: `null`")
    else:
        lines.append(f"- `score`: `{fmt_num(score)}`")

    lines.append(f"- `n_flags`: `{run.get('n_flags', 0)}`")

    if "api_failure" in run:
        lines.append(f"- `api_failure`: `{run.get('api_failure')}`")
    if "n_agent_errors" in run:
        lines.append(f"- `n_agent_errors`: `{run.get('n_agent_errors')}`")
    if run.get("error"):
        lines.append(f"- `error`: `{run['error']}`")

    warnings = run.get("warnings", [])
    if warnings:
        lines.append("- `warnings`:")
        for warning in warnings:
            lines.append(f"  - {warning}")

    flags = run.get("flags", [])
    if flags:
        lines.append("- `flags`:")
        for flag in flags:
            snippet = flag.get("flagged_text", "").replace("\n", " ").strip()
            lines.append(
                "  - "
                f"{flag.get('agent', 'UNKNOWN')} · "
                f"{flag.get('bias_type', 'unknown')} · "
                f"{flag.get('severity', 'unknown')} · "
                f"conf `{flag.get('confidence', 0):.3f}` · "
                f"\"{snippet}\""
            )
    else:
        lines.append("- `flags`: none")

    lines.append("")
    return lines


def main() -> None:
    leaderboard = json.loads(LEADERBOARD.read_text())
    rerun = json.loads(RERUN.read_text()) if RERUN.exists() else None
    results = sorted(leaderboard.get("results", []), key=sortkey)

    lines: list[str] = []
    lines.append("# BiasScan Detailed Leaderboard Report")
    lines.append("")
    lines.append("## Canonical storage")
    lines.append("")
    lines.append(
        f"- Latest merged benchmark artifact: `{LEADERBOARD.relative_to(ROOT)}` "
        f"(modified {fmt_ts(LEADERBOARD)})"
    )
    if rerun:
        lines.append(
            f"- Rerun-only artifact: `{RERUN.relative_to(ROOT)}` "
            f"(modified {fmt_ts(RERUN)})"
        )
    lines.append(
        "- This benchmark's per-run detail is stored inline under "
        "`results[*].papers[*].runs[*]` in `leaderboard.json`."
    )
    lines.append(
        "- There is no deeper persisted per-agent trace archive for this specific "
        "benchmark in `eval/results/`; that directory is empty for the latest run."
    )
    lines.append(
        "- `docs/MODEL_EVALUATION.md` is a narrative write-up, not the canonical latest raw data file."
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Papers: `{', '.join(leaderboard.get('papers', []))}`")
    lines.append(f"- Runs per paper in the stored artifact: `{leaderboard.get('n_runs')}`")
    lines.append(f"- Models in the latest merged leaderboard: `{len(results)}`")
    lines.append(
        "- Warning-level metadata such as `api_failure`, `n_agent_errors`, and `warnings` "
        "exists only for the five rerun models that were merged back in from `leaderboard_rerun.json`."
    )
    lines.append("")
    lines.append("## Summary leaderboard")
    lines.append("")
    lines.append(
        "| Rank | Model | Tier | Nu-OG | Nu-OG-GPT | Nu-Edit | Nu-bias-injected | Status |"
    )
    lines.append(
        "|---:|---|---|---:|---:|---:|---:|---|"
    )

    rank = 0
    for result in results:
        result_status = status(result)
        if result_status == "OK":
            rank += 1
            rank_cell = str(rank)
        else:
            rank_cell = "·"

        cells = []
        for paper in PAPERS_ORDER:
            paper_data = result.get("papers", {}).get(paper, {})
            if paper_data:
                cells.append(f"{fmt_num(paper_data.get('mean', 0.0))}±{fmt_num(paper_data.get('sd', 0.0))}")
            else:
                cells.append("—")
        lines.append(
            f"| {rank_cell} | `{result['model']}` | {result['tier']} | "
            f"{cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {result_status} |"
        )

    lines.append("")
    lines.append("## Per-model full detail")
    lines.append("")

    for result in results:
        model = result["model"]
        result_status = status(result)
        lines.append(f"### {model}")
        lines.append("")
        lines.append(f"- Tier: `{result['tier']}`")
        lines.append(f"- Status: `{result_status}`")
        if result.get("crashed"):
            lines.append(f"- Crash: `{result['crashed']}`")
        lines.append("")
        lines.append("| Paper | Mean | SD | Valid runs | Elapsed (s) |")
        lines.append("|---|---:|---:|---:|---:|")
        for paper in PAPERS_ORDER:
            paper_data = result.get("papers", {}).get(paper, {})
            if not paper_data:
                lines.append(f"| {paper} | — | — | — | — |")
                continue
            lines.append(
                f"| {paper} | {fmt_num(paper_data.get('mean', 0.0))} | "
                f"{fmt_num(paper_data.get('sd', 0.0))} | "
                f"{paper_data.get('valid', 0)} | "
                f"{fmt_num(paper_data.get('elapsed_s', 0.0))} |"
            )
        lines.append("")

        for paper in PAPERS_ORDER:
            paper_data = result.get("papers", {}).get(paper, {})
            if not paper_data:
                continue
            lines.append(f"#### {paper}")
            lines.append("")
            lines.append(
                f"- Aggregate: mean `{fmt_num(paper_data.get('mean', 0.0))}`, "
                f"sd `{fmt_num(paper_data.get('sd', 0.0))}`, "
                f"valid `{paper_data.get('valid', 0)}`, "
                f"elapsed `{fmt_num(paper_data.get('elapsed_s', 0.0))}` seconds"
            )
            if "api_failures" in paper_data:
                lines.append(f"- `api_failures`: `{paper_data.get('api_failures', 0)}`")
            lines.append("")
            for idx, run in enumerate(paper_data.get("runs", []), start=1):
                lines.extend(summarize_run(run, idx))

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
