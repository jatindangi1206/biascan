"""V2-prompt sentence-level eval — 1 run per (sentence × model).

Same 48 sentences and same 5-model panel as sentence_eval.py, but uses the v2
prompts via analysis_mode='general_research' and only 1 run per cell (= 240
analyze() calls, ~$1).

Outputs:
  eval/output/sentence_eval_v2_per_run.csv     one row per (sentence × model)
  eval/output/sentence_eval_v2_summary.csv     same shape (1 run = the row)
  eval/output/sentence_eval_v2_progress.json
"""
from __future__ import annotations
import asyncio, csv, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.orchestrator import Orchestrator  # noqa: E402
from app.providers.base import ProviderConfig  # noqa: E402

DATASET = ROOT / "backend" / "app" / "sentences-dataset" / "dataset .csv"
OUT_DIR = ROOT / "eval" / "output"
PER_RUN_CSV = OUT_DIR / "sentence_eval_v2_per_run.csv"
PROGRESS_JSON = OUT_DIR / "sentence_eval_v2_progress.json"

N_RUNS = 1
ANALYSIS_MODE = "general_research"  # → v2 prompts

MODELS: list[dict] = [
    {"id": "openai/gpt-4.1-nano",                  "company": "OpenAI"},
    {"id": "anthropic/claude-3-haiku",             "company": "Anthropic"},
    {"id": "google/gemini-2.5-flash-lite",         "company": "Google"},
    {"id": "deepseek/deepseek-v4-flash",           "company": "DeepSeek"},
    {"id": "meta-llama/llama-3.3-70b-instruct",    "company": "Meta"},
]

LABEL_MAP = {
    "Confirmation Bias":   "confirmation_bias",
    "Certainty Inflation": "certainty_inflation",
    "Framing Effects":     "framing_effect",
    "Causal Inference":    "causal_inference_error",
    "Overgeneralization":  "overgeneralisation",
}


def load_dataset() -> list[dict]:
    rows = []
    with DATASET.open() as f:
        for row in csv.DictReader(f):
            label = row["Annotated Bias"].strip()
            expected = LABEL_MAP.get(label)
            if expected is None:
                print(f"  ! unknown label {label!r} for row {row['Sentence ID']}, skipping")
                continue
            rows.append({
                "sentence_id": row["Sentence ID"].strip(),
                "sentence": row["Sentence Text"].strip(),
                "expected_label_human": label,
                "expected_bias_type": expected,
            })
    return rows


def short(s: str, n: int = 220) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


async def one_run(orch: Orchestrator, sentence: str, cfg: ProviderConfig) -> dict:
    try:
        resp = await orch.analyze(
            text=sentence,
            references=None,
            mode="lite",
            analysis_mode=ANALYSIS_MODE,
            provider_config=cfg,
            agents=None,
        )
        anns = resp.annotations
        agents = sorted({a.agent_name for a in anns})
        bias_types = sorted({a.bias_type for a in anns})
        severities = [a.severity for a in anns]
        confidences = [round(a.confidence, 2) for a in anns]
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


async def main() -> int:
    api_key = os.environ["BIASSCAN_API_KEY"]
    rows = load_dataset()
    print(f"Loaded {len(rows)} sentences from {DATASET}")
    print(f"V2 prompts via analysis_mode='{ANALYSIS_MODE}'")
    print(f"Plan: {len(rows)} sentences × {len(MODELS)} models × {N_RUNS} run = "
          f"{len(rows) * len(MODELS) * N_RUNS} analyze() calls\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    orch = Orchestrator()
    started = time.time()

    per_run_cols = [
        "sentence_id", "sentence", "expected_label_human", "expected_bias_type",
        "company", "model", "run",
        "score", "n_flags", "agents_fired", "bias_types_detected",
        "severities", "confidences", "correct", "reasoning",
        "api_errors", "ok", "error",
    ]
    PER_RUN_CSV.write_text(",".join(per_run_cols) + "\n")

    def append_per_run(d: dict) -> None:
        with PER_RUN_CSV.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_run_cols, extrasaction="ignore")
            w.writerow(d)

    for m_idx, mm in enumerate(MODELS, 1):
        model = mm["id"]
        cfg = ProviderConfig(provider="openrouter", model=model, api_key=api_key, base_url=None)
        print(f"\n{'#'*80}\n# [{m_idx}/{len(MODELS)}] {mm['company']:<10} · {model}\n{'#'*80}",
              flush=True)
        for s_idx, row in enumerate(rows, 1):
            sentence = row["sentence"]
            rr = await one_run(orch, sentence, cfg)
            detected_set = set(rr.get("bias_types_detected", "").split("|"))
            detected_set.discard("")
            correct = row["expected_bias_type"] in detected_set
            print(f"  [{s_idx:>2}/{len(rows)}] id={row['sentence_id']:<3} "
                  f"({row['expected_bias_type'][:18]:<18}) "
                  f"score={rr.get('score')} n_flags={rr['n_flags']} "
                  f"agents={rr['agents_fired']} correct={int(correct)}",
                  flush=True)
            append_per_run({
                "sentence_id": row["sentence_id"],
                "sentence": short(row["sentence"], 280),
                "expected_label_human": row["expected_label_human"],
                "expected_bias_type": row["expected_bias_type"],
                "company": mm["company"],
                "model": model,
                "run": 1,
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
            PROGRESS_JSON.write_text(json.dumps({
                "completed_model": m_idx, "completed_sentence_in_model": s_idx,
                "elapsed_s": round(time.time() - started, 1),
            }, indent=2))

    print(f"\nDone. Total elapsed: {time.time()-started:.0f}s")
    print(f"Per-run CSV:  {PER_RUN_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
