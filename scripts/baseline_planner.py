#!/usr/bin/env python
"""Baseline the DSPy Planner against CodeOrch's 10-task eval set.

Run this BEFORE Day 9's MIPRO optimization so we have an honest "before"
number for the before/after benchmark chart.

Two metric paths supported:

  --metric=validity (Day 8 default) — deterministic schema-level scoring.
                                       Cheap, but Sonnet hits 1.0 on every
                                       spec, leaving MIPRO no headroom.
  --metric=judge    (Day 9)         — LLM-as-judge composite: spec coverage,
                                       task atomicity, AC testability. Real
                                       gradient at the cost of judge calls.

Output filename is auto-stamped by (model, metric) so Day 8 evidence and
Day 9 baselines coexist on disk:
    reports/baseline_validity_sonnet.json   (== Day 8 baseline_planner.json)
    reports/baseline_judge_sonnet.json
    reports/baseline_judge_haiku.json       (the Day 9 demoted baseline)

Usage:
    # Day 9 — Haiku-as-Planner with LLM-as-judge metric
    python scripts/baseline_planner.py --model haiku --metric judge

    # Sonnet baseline with new metric (sanity check; should land ~0.83)
    python scripts/baseline_planner.py --model sonnet --metric judge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
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
    PLANNER_MODEL_ID,
    configure_lm,
    plan_validity_score,
)
from dspy_modules.judge import score_with_breakdown  # noqa: E402


# Single source of truth for the eval set.
PROMPTFOO_YAML = ROOT.parent / "codeorch" / "evals" / "promptfoo.yaml"


def load_specs() -> list[dict]:
    if not PROMPTFOO_YAML.exists():
        raise FileNotFoundError(f"Cannot find {PROMPTFOO_YAML} — codeorch must be a sibling dir.")
    data = yaml.safe_load(PROMPTFOO_YAML.read_text())
    return [
        {
            "id": t["description"],
            "spec": t["vars"]["spec"],
            "difficulty": t.get("metadata", {}).get("difficulty", "unknown"),
            "language": t.get("metadata", {}).get("language", "unknown"),
        }
        for t in data["tests"]
    ]


def _resolve_model(name: str) -> str:
    if name == "sonnet":
        return PLANNER_MODEL_ID
    if name == "haiku":
        return HAIKU_MODEL_ID
    return name  # let user pass full IDs through


def _default_out(model_name: str, metric: str) -> Path:
    short = "sonnet" if "sonnet" in model_name else "haiku" if "haiku" in model_name else "model"
    return ROOT / "reports" / f"baseline_{metric}_{short}.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sonnet", help="sonnet | haiku | full DSPy model ID")
    p.add_argument("--metric", choices=["validity", "judge"], default="judge",
                   help="validity = Day 8 deterministic; judge = Day 9 LLM-as-judge composite")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path. Default: reports/baseline_{metric}_{model}.json")
    p.add_argument("--limit", type=int, default=0,
                   help="if >0, only run first N specs (debug)")
    args = p.parse_args()

    model_id = _resolve_model(args.model)
    out_path = args.out or _default_out(model_id, args.metric)

    configure_lm(model_id=model_id)
    planner = DSPyPlanner()
    specs = load_specs()
    if args.limit:
        specs = specs[: args.limit]

    print(f"Baselining {model_id} on {len(specs)} specs (metric={args.metric}) ...")
    print()

    per_spec = []
    started = time.monotonic()
    for s in specs:
        t0 = time.monotonic()
        record = {
            "id": s["id"], "difficulty": s["difficulty"], "language": s["language"],
            "ok": False, "score": 0.0, "n_tasks": 0, "error": None, "latency_ms": 0,
        }
        try:
            pred = planner(spec=s["spec"])
            example = dspy.Example(spec=s["spec"])
            if args.metric == "validity":
                record["score"] = plan_validity_score(example, pred)
                record["n_tasks"] = len(pred.plan.tasks)
                record["ok"] = True
            else:  # judge
                breakdown = score_with_breakdown(example, pred)
                record["score"] = breakdown["composite"]
                record["n_tasks"] = len(pred.plan.tasks)
                record["ok"] = breakdown["ok"]
                record["spec_coverage"] = breakdown.get("spec_coverage")
                record["task_atomicity"] = breakdown.get("task_atomicity")
                record["acceptance_testability"] = breakdown.get("acceptance_testability")
                record["rationales"] = breakdown.get("rationales")
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        record["latency_ms"] = int((time.monotonic() - t0) * 1000)
        per_spec.append(record)

        status = "PASS" if record["ok"] else "FAIL"
        bits = f"score={record['score']:.2f}"
        if args.metric == "judge" and record.get("spec_coverage") is not None:
            bits += (f" cov={record['spec_coverage']:.2f}"
                     f" atom={record['task_atomicity']:.2f}"
                     f" test={record['acceptance_testability']:.2f}")
        bits += f" tasks={record['n_tasks']} {record['latency_ms']} ms"
        print(f"  [{status}] {s['id']:<35} {bits}")

    total_ms = int((time.monotonic() - started) * 1000)
    mean_score = sum(r["score"] for r in per_spec) / max(len(per_spec), 1)
    pass_rate = sum(1 for r in per_spec if r["ok"]) / max(len(per_spec), 1)

    out = {
        "model": model_id,
        "metric": args.metric,
        "n_specs": len(per_spec),
        "mean_score": round(mean_score, 4),
        "pass_rate": round(pass_rate, 4),
        "total_latency_ms": total_ms,
        "per_spec": per_spec,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print()
    print("=" * 60)
    print(f"model: {model_id}    metric: {args.metric}")
    print(f"mean_score: {mean_score:.4f}    pass_rate: {pass_rate:.0%}    total: {total_ms} ms")
    print(f"Wrote {out_path}")
    return 0 if pass_rate > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
