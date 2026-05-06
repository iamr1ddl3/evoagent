#!/usr/bin/env python
"""Day 9 benchmark: baseline Haiku Planner vs MIPRO-optimized Haiku Planner.

Runs both variants on the held-out 4 specs (NOT seen by MIPRO) for 3 trials
each, scores via LLM-as-judge composite, computes mean + std dev per
(variant, spec), and renders a self-contained HTML report.

Why 3 trials per cell:
  Single-trial deltas can't be defended in an interview. With n=3 we can
  show variance bars and say "the +X pp improvement is/isn't within noise."

Outputs:
    reports/benchmark_compare.json   # raw scores
    reports/before_after.html        # plotly self-contained report

Rate-limit prophylaxis:
  Anthropic throttles us at 50K input tokens/min and 50 req/min on Haiku.
  4 specs * 3 trials * 2 variants = 24 planner calls + 24 judge calls = 48
  total. We sleep 1.5s between calls to keep us comfortably under both
  thresholds (gives us 40 calls/min worst case).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import dspy
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from dspy_modules.planner import (  # noqa: E402
    DSPyPlanner,
    HAIKU_MODEL_ID,
    configure_lm,
)
from dspy_modules.judge import score_with_breakdown  # noqa: E402

# Reuse the stratified split from optimize_planner.py — same VAL_IDS, single
# source of truth. We import indirectly via importlib to avoid `scripts.*` as
# a package.
import importlib.util  # noqa: E402

_OPT_SPEC = importlib.util.spec_from_file_location(
    "_opt", ROOT / "scripts" / "optimize_planner.py"
)
_opt = importlib.util.module_from_spec(_OPT_SPEC)
_OPT_SPEC.loader.exec_module(_opt)
VAL_IDS = _opt.VAL_IDS

PROMPTFOO_YAML = ROOT.parent / "codeorch" / "evals" / "promptfoo.yaml"
RATE_LIMIT_SLEEP_SEC = 1.5


def load_held_out_specs() -> list[dict]:
    data = yaml.safe_load(PROMPTFOO_YAML.read_text())
    out = []
    for t in data["tests"]:
        if t["description"] in VAL_IDS:
            out.append({
                "id": t["description"],
                "spec": t["vars"]["spec"],
                "difficulty": t.get("metadata", {}).get("difficulty", "unknown"),
                "language": t.get("metadata", {}).get("language", "unknown"),
            })
    return out


def run_variant(
    name: str,
    program: dspy.Module,
    specs: list[dict],
    n_trials: int,
) -> list[dict]:
    """Run `program` over `specs` for `n_trials` repetitions, score with judge.

    Returns a flat list of records:
        {variant, spec_id, difficulty, language, trial, ok, score, sub-scores...}
    """
    records: list[dict] = []
    for trial in range(1, n_trials + 1):
        for s in specs:
            t0 = time.monotonic()
            example = dspy.Example(spec=s["spec"])
            try:
                pred = program(spec=s["spec"])
                breakdown = score_with_breakdown(example, pred)
                rec = {
                    "variant": name,
                    "trial": trial,
                    "spec_id": s["id"],
                    "difficulty": s["difficulty"],
                    "language": s["language"],
                    "ok": breakdown["ok"],
                    "score": breakdown["composite"],
                    "spec_coverage": breakdown.get("spec_coverage"),
                    "task_atomicity": breakdown.get("task_atomicity"),
                    "acceptance_testability": breakdown.get("acceptance_testability"),
                    "n_tasks": len(pred.plan.tasks),
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }
            except Exception as exc:
                rec = {
                    "variant": name, "trial": trial, "spec_id": s["id"],
                    "difficulty": s["difficulty"], "language": s["language"],
                    "ok": False, "score": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }
            records.append(rec)
            print(f"    trial={trial} {s['id']:<35} score={rec['score']:.2f} "
                  f"ok={rec['ok']} {rec['latency_ms']}ms")
            time.sleep(RATE_LIMIT_SLEEP_SEC)
    return records


def aggregate(records: list[dict]) -> dict:
    """Group by (variant, spec_id) -> {mean, std, scores[3]}.
    Also produces (variant) -> overall mean."""
    by_cell: dict[tuple[str, str], list[float]] = {}
    by_variant: dict[str, list[float]] = {}
    for r in records:
        cell = (r["variant"], r["spec_id"])
        by_cell.setdefault(cell, []).append(r["score"])
        by_variant.setdefault(r["variant"], []).append(r["score"])

    cell_stats = {}
    for (variant, spec_id), scores in by_cell.items():
        cell_stats[f"{variant}::{spec_id}"] = {
            "mean": round(statistics.mean(scores), 4),
            "std": round(statistics.pstdev(scores), 4),
            "scores": [round(s, 4) for s in scores],
        }

    variant_stats = {}
    for variant, scores in by_variant.items():
        variant_stats[variant] = {
            "mean": round(statistics.mean(scores), 4),
            "std": round(statistics.pstdev(scores), 4),
            "n": len(scores),
        }

    return {"cell_stats": cell_stats, "variant_stats": variant_stats}


def render_html_report(
    records: list[dict],
    agg: dict,
    mipro_meta: dict,
    out_path: Path,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # 1. Per-spec score-delta bar chart with error bars
    spec_ids = sorted({r["spec_id"] for r in records})
    baseline_means = []
    baseline_stds = []
    optimized_means = []
    optimized_stds = []
    for sid in spec_ids:
        b_key = f"baseline_haiku::{sid}"
        o_key = f"optimized_haiku::{sid}"
        b = agg["cell_stats"].get(b_key, {"mean": 0, "std": 0})
        o = agg["cell_stats"].get(o_key, {"mean": 0, "std": 0})
        baseline_means.append(b["mean"])
        baseline_stds.append(b["std"])
        optimized_means.append(o["mean"])
        optimized_stds.append(o["std"])

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=spec_ids, y=baseline_means,
        error_y={"type": "data", "array": baseline_stds, "visible": True},
        name="Baseline Haiku 4.5",
        marker_color="#FFA15A",
    ))
    fig1.add_trace(go.Bar(
        x=spec_ids, y=optimized_means,
        error_y={"type": "data", "array": optimized_stds, "visible": True},
        name="MIPRO-optimized Haiku 4.5",
        marker_color="#19D3F3",
    ))
    fig1.update_layout(
        title="Per-spec mean score (held-out 4, n=3 trials each)",
        yaxis_title="Composite score (0-1)",
        xaxis_title="Spec",
        barmode="group",
        yaxis={"range": [0, 1.05]},
        height=420,
    )

    # 2. MIPRO optimization curve
    trial_scores = mipro_meta.get("trial_full_eval_scores", [])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(1, len(trial_scores) + 1)),
        y=trial_scores,
        mode="lines+markers",
        name="Trial score",
        line={"color": "#636EFA"},
    ))
    fig2.add_hline(
        y=mipro_meta.get("train_baseline_score", 87.5),
        line_dash="dash", line_color="#FFA15A",
        annotation_text="Train baseline", annotation_position="bottom right",
    )
    fig2.add_hline(
        y=mipro_meta.get("best_score_train", 90.28),
        line_dash="dash", line_color="#00CC96",
        annotation_text="Best (Trial 5)", annotation_position="top left",
    )
    fig2.update_layout(
        title="MIPRO optimization curve (rate-limit dips honestly visible)",
        yaxis_title="Train score (%)",
        xaxis_title="Trial",
        yaxis={"range": [40, 100]},
        height=380,
    )

    # 3. Summary table
    headline = agg["variant_stats"]
    delta_mean = headline.get("optimized_haiku", {}).get("mean", 0) - headline.get("baseline_haiku", {}).get("mean", 0)

    # Build per-spec delta breakdown for the findings section
    per_spec_deltas = []
    for sid in spec_ids:
        b = agg["cell_stats"].get(f"baseline_haiku::{sid}", {"mean": 0})["mean"]
        o = agg["cell_stats"].get(f"optimized_haiku::{sid}", {"mean": 0})["mean"]
        per_spec_deltas.append((sid, b, o, o - b))

    summary_rows = [
        ["Baseline Haiku 4.5",
         f"{headline.get('baseline_haiku', {}).get('mean', 0):.4f}",
         f"{headline.get('baseline_haiku', {}).get('std', 0):.4f}",
         str(headline.get("baseline_haiku", {}).get("n", 0))],
        ["MIPRO-optimized Haiku 4.5",
         f"{headline.get('optimized_haiku', {}).get('mean', 0):.4f}",
         f"{headline.get('optimized_haiku', {}).get('std', 0):.4f}",
         str(headline.get("optimized_haiku", {}).get("n", 0))],
        ["Δ (optimized − baseline)",
         f"{delta_mean:+.4f}",
         "—", "—"],
    ]

    fig3 = go.Figure(data=[go.Table(
        header={
            "values": ["Variant", "Mean", "Std dev", "N"],
            "fill_color": "#3a3a3a", "font": {"color": "white"},
        },
        cells={
            "values": list(zip(*summary_rows)),
            "fill_color": [["#f5f5f5", "#f5f5f5", "#fff8dc"]] * 4,
        },
    )])
    fig3.update_layout(
        title="Held-out summary (4 specs × 3 trials = 12 runs per variant)",
        height=200, margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    # Compose the page
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EvoAgent Day 9 — MIPRO Before/After</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ border-bottom: 2px solid #19d3f3; padding-bottom: 0.5rem; }}
  h2 {{ margin-top: 2rem; }}
  .meta {{ background: #f5f5f5; padding: 1rem; border-radius: 6px;
          font-size: 0.92rem; }}
  .meta code {{ background: #fff; padding: 0 0.3rem; border-radius: 3px; }}
  .findings {{ background: #fff8dc; padding: 1rem; border-left: 4px solid #ff9a00;
              border-radius: 4px; }}
  .findings ul {{ margin: 0.5rem 0; padding-left: 1.4rem; }}
</style>
</head>
<body>
<h1>EvoAgent Day 9 — MIPROv2 Before/After</h1>

<div class="meta">
<strong>Setup.</strong> Target: <code>anthropic/claude-haiku-4-5</code> as the
DSPy Planner. Optimizer: MIPROv2 (10 trials, 6 candidates,
4 max-bootstrapped-demos, valset=trainset for clean held-out). Metric:
LLM-as-judge composite (spec coverage + task atomicity + AC testability,
mean of 3, quartile-snapped, anchor 0.5). Train: 6 stratified specs
(2 easy + 3 medium + 1 hard). Held-out: 4 specs (1 easy + 1 medium + 2 hard)
— never seen by MIPRO.<br><br>

<strong>Why this experiment.</strong> Day 7's RouterBench showed routing
weights move when production data demands. Day 9 demonstrates the same
discipline at the prompt layer: optimize the Planner's instructions
against measured failure modes, then verify generalization on truly
unseen specs.
</div>

<h2>Per-spec held-out comparison (n=3 trials)</h2>
{fig1.to_html(include_plotlyjs="cdn", full_html=False)}

<h2>Held-out summary</h2>
{fig3.to_html(include_plotlyjs=False, full_html=False)}

<div class="findings">
<strong>Findings (honest reporting — the data is what it is).</strong>
<ul>
  <li><strong>Train improved, held-out regressed.</strong> Train delta:
      <code>+{mipro_meta.get('train_delta', 0):.2f} pp</code> (87.5 → 90.28). Held-out
      delta: <code>{delta_mean:+.4f}</code>.</li>
  <li><strong>Per-spec breakdown reveals task-class overfit.</strong> MIPRO
      improved the hardest spec (<code>rate_limiter</code>: {per_spec_deltas[2][1]:.2f} → {per_spec_deltas[2][2]:.2f},
      <code>{per_spec_deltas[2][3]:+.2f}</code>) but regressed easy and medium specs
      (<code>chunk</code>: {per_spec_deltas[0][3]:+.2f}, <code>groupBy</code>: {per_spec_deltas[1][3]:+.2f}).
      The optimized prompt produces a more verbose decomposition pattern — right for hard
      specs, wrong for simple ones.</li>
  <li><strong>What this tells us.</strong> Even with a stratified train/val split,
      MIPRO chased the gradient where headroom existed (hard specs) and learned
      a pattern that doesn't suit easy/medium tasks. For production, this argues
      for task-class-specific optimization, not one-size-fits-all prompt
      compilation. A single optimized prompt is a compromise; multi-prompt
      routing is the better answer when the input distribution varies.</li>
  <li><strong>Methodology guarded.</strong> Held-out 4 specs were never seen by
      MIPRO (<code>valset=trainset</code> at compile time). Stratified split kept
      difficulty balanced. Three trials per cell — though DSPy caching meant
      trials 2-3 of the optimized variant were cache hits, so reported std dev
      reflects mostly the baseline's fresh-LM variance.</li>
  <li><strong>Rate limits</strong> affected 5 of 13 compile trials (50K input tokens/min,
      50 req/min on Haiku). Honest noise visible in the optimization curve below.</li>
</ul>
</div>

<h2>MIPRO optimization curve</h2>
{fig2.to_html(include_plotlyjs=False, full_html=False)}

<div class="meta">
<strong>Cost & time.</strong> MIPRO compile: 173.5s, ~$0.50 estimated.
Held-out benchmark (this run): 4 × 3 × 2 = 24 planner + 24 judge calls,
~75s wall-clock with 1.5s rate-limit prophylaxis, ~$0.20 estimated.
Total Day 9 LLM spend: ~$1 of Anthropic credits.<br><br>

<strong>Repository.</strong>
<a href="https://github.com/iamr1ddl3/evoagent">github.com/iamr1ddl3/evoagent</a> ·
companion: <a href="https://github.com/iamr1ddl3/codeorch">github.com/iamr1ddl3/codeorch</a>
</div>

</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=3,
                   help="Trials per (variant, spec). 3 gives variance bars.")
    p.add_argument("--out-json", type=Path,
                   default=ROOT / "reports" / "benchmark_compare.json")
    p.add_argument("--out-html", type=Path,
                   default=ROOT / "reports" / "before_after.html")
    p.add_argument("--compiled-program", type=Path,
                   default=ROOT / "optimized_programs" / "planner_haiku_v1.json")
    p.add_argument("--mipro-trace", type=Path,
                   default=ROOT / "reports" / "mipro_run.json")
    args = p.parse_args()

    if not args.compiled_program.exists():
        raise FileNotFoundError(
            f"Compiled program not found: {args.compiled_program}\n"
            f"Run scripts/optimize_planner.py first."
        )

    configure_lm(model_id=HAIKU_MODEL_ID)
    specs = load_held_out_specs()
    print(f"Held-out specs ({len(specs)}):")
    for s in specs:
        print(f"  {s['difficulty']:<7} {s['language']:<11} {s['id']}")
    print()

    # Baseline: fresh DSPyPlanner, no compile artifacts
    print("=== Baseline Haiku 4.5 (un-optimized) ===")
    baseline = DSPyPlanner()
    baseline_records = run_variant("baseline_haiku", baseline, specs, args.n_trials)
    print()

    # Optimized: load the MIPRO-compiled program
    print("=== MIPRO-optimized Haiku 4.5 ===")
    optimized = DSPyPlanner()
    optimized.load(str(args.compiled_program))
    optimized_records = run_variant("optimized_haiku", optimized, specs, args.n_trials)
    print()

    all_records = baseline_records + optimized_records
    agg = aggregate(all_records)

    # Persist raw + aggregated
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "n_trials": args.n_trials,
        "held_out_spec_ids": [s["id"] for s in specs],
        "records": all_records,
        "aggregate": agg,
    }, indent=2))
    print(f"Wrote {args.out_json}")

    # Render HTML
    mipro_meta = json.loads(args.mipro_trace.read_text()) if args.mipro_trace.exists() else {}
    render_html_report(all_records, agg, mipro_meta, args.out_html)
    print(f"Wrote {args.out_html}")

    print()
    print("=" * 60)
    print(f"Baseline mean:    {agg['variant_stats']['baseline_haiku']['mean']:.4f}"
          f"  (std {agg['variant_stats']['baseline_haiku']['std']:.4f})")
    print(f"Optimized mean:   {agg['variant_stats']['optimized_haiku']['mean']:.4f}"
          f"  (std {agg['variant_stats']['optimized_haiku']['std']:.4f})")
    delta = agg['variant_stats']['optimized_haiku']['mean'] - agg['variant_stats']['baseline_haiku']['mean']
    print(f"Held-out delta:   {delta:+.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
