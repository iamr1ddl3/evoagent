#!/usr/bin/env python
"""Run MIPROv2 to optimize the DSPy Planner against the LLM-as-judge metric.

Day 9 deliverable. Reads codeorch's promptfoo.yaml (single source of truth),
splits 6-train / 4-held-out (reproducibly, by file order — not shuffled),
runs MIPROv2.compile against the judge metric, persists the optimized program.

Budget guards (explicit, NOT defaults):
  num_trials             = 10  (vs. MIPRO default ~30-50)
  num_candidates         = 6   (vs. default 10)
  max_bootstrapped_demos = 4   (vs. default 5)
  max_labeled_demos      = 0   (we don't have labeled gold plans)

Estimated spend at these settings: ~$0.50-$0.80. Wall clock ~25-45 min.
We use Haiku as the target LM (cheaper trials) and Sonnet as the judge
(quality scoring) — explicit cost-quality split.

Outputs:
    optimized_programs/planner_haiku_v1.json   (compiled DSPy program)
    reports/mipro_run.json                      (per-trial scores + meta)

Usage:
    python scripts/optimize_planner.py
    python scripts/optimize_planner.py --num-trials 5    # smaller pilot run
"""

from __future__ import annotations

import argparse
import json
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
from dspy_modules.judge import composite_plan_score  # noqa: E402


PROMPTFOO_YAML = ROOT.parent / "codeorch" / "evals" / "promptfoo.yaml"

# STRATIFIED 6-train / 4-val split (NOT random shuffle, NOT file-order).
#
# Reasoning: file-order split puts 3 easy + 3 medium in train, 1 medium +
# 3 hard in val — train/val have different difficulty distributions and
# the benchmark would conflate optimization gains with distribution shift.
#
# Stratified split (chosen below):
#   Train: 2 easy + 3 medium + 1 hard (the easier hard one: topological_sort)
#   Val:   1 easy + 1 medium + 2 hard (rate_limiter, event_emitter)
# This keeps difficulty mix similar and reserves the two hardest specs
# for held-out evaluation — MIPRO has to generalize to truly novel patterns.
TRAIN_IDS = {
    "easy/python/clamp",
    "easy/python/is_palindrome",
    "medium/python/lru_cache",
    "medium/typescript/debounce",       # this one scored 0.67 in baseline
    "medium/python/flatten",
    "hard/python/topological_sort",     # easier hard
}
VAL_IDS = {
    "easy/typescript/chunk",
    "medium/javascript/groupBy",
    "hard/python/rate_limiter",         # this one scored 0.67 in baseline
    "hard/typescript/event_emitter",
}


def load_examples() -> tuple[list[dspy.Example], list[dspy.Example], list[dict]]:
    """Return (train_examples, val_examples, all_specs_meta).

    Stratified split — see TRAIN_IDS / VAL_IDS above. The split is committed
    to source so the held-out 4 are reproducible: any reader can verify the
    Day 9 results by re-running with the same code.
    """
    if not PROMPTFOO_YAML.exists():
        raise FileNotFoundError(f"Cannot find {PROMPTFOO_YAML}.")

    data = yaml.safe_load(PROMPTFOO_YAML.read_text())
    specs = []
    for t in data["tests"]:
        specs.append({
            "id": t["description"],
            "spec": t["vars"]["spec"],
            "difficulty": t.get("metadata", {}).get("difficulty", "unknown"),
            "language": t.get("metadata", {}).get("language", "unknown"),
        })

    seen = {s["id"] for s in specs}
    missing_train = TRAIN_IDS - seen
    missing_val = VAL_IDS - seen
    if missing_train or missing_val:
        raise RuntimeError(
            f"Spec IDs from split don't match promptfoo.yaml. "
            f"Missing in train: {missing_train}; missing in val: {missing_val}"
        )

    train_meta = [s for s in specs if s["id"] in TRAIN_IDS]
    val_meta = [s for s in specs if s["id"] in VAL_IDS]

    train_examples = [
        dspy.Example(spec=s["spec"]).with_inputs("spec") for s in train_meta
    ]
    val_examples = [
        dspy.Example(spec=s["spec"]).with_inputs("spec") for s in val_meta
    ]
    return train_examples, val_examples, specs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--num-candidates", type=int, default=6)
    p.add_argument("--max-bootstrapped-demos", type=int, default=4)
    p.add_argument("--out-program",
                   type=Path,
                   default=ROOT / "optimized_programs" / "planner_haiku_v1.json")
    p.add_argument("--out-trace",
                   type=Path,
                   default=ROOT / "reports" / "mipro_run.json")
    args = p.parse_args()

    # Configure Haiku as the optimization target
    configure_lm(model_id=HAIKU_MODEL_ID)

    train, val, all_specs = load_examples()
    print(f"Train set: {len(train)} specs (stratified — see TRAIN_IDS)")
    print(f"Held-out set: {len(val)} specs (reserved for Day 9 benchmark)")
    print()

    print("Estimated MIPRO budget:")
    n_trials = args.num_trials
    print(f"  ~{n_trials} trials × {len(train)} train specs = ~{n_trials * len(train)} Haiku planner calls")
    print(f"  ~{n_trials * len(train)} judge calls (Sonnet 4.6, cached on rerun)")
    print(f"  + {args.num_candidates} candidate-instruction generations (Sonnet)")
    print(f"  Estimated cost: $0.50-$0.80")
    print()

    student = DSPyPlanner()

    # The optimizer instantiation
    from dspy.teleprompt import MIPROv2  # noqa: WPS433
    optimizer = MIPROv2(
        metric=composite_plan_score,
        num_candidates=args.num_candidates,
        # auto=None disables MIPRO's auto-budget heuristic; we set budget ourselves
        auto=None,
        init_temperature=1.0,
    )

    print(f"Compiling MIPROv2 (num_trials={n_trials}, "
          f"num_candidates={args.num_candidates}, "
          f"max_bootstrapped_demos={args.max_bootstrapped_demos}) ...")
    print()
    print("Note on validation methodology:")
    print("  MIPROv2.compile() requires an internal validation set during")
    print("  optimization. We pass `valset=trainset` so the held-out 4 specs")
    print("  remain TRULY UNSEEN by MIPRO. This means MIPRO may slightly")
    print("  overfit to train, but the post-compile benchmark on val stays")
    print("  uncontaminated — exactly what we need for an honest delta.")
    print()
    t0 = time.monotonic()
    compiled = optimizer.compile(
        student=student,
        trainset=train,
        valset=train,                               # see methodology note above
        num_trials=n_trials,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=0,                         # no labeled gold plans
        minibatch_size=len(train),                   # use full train each pass
        requires_permission_to_run=False,
    )
    elapsed_sec = time.monotonic() - t0
    print()
    print(f"Compile finished in {elapsed_sec:.1f}s")

    # Persist the compiled program
    args.out_program.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(args.out_program))
    print(f"Wrote compiled program: {args.out_program}")

    # Save run metadata + per-trial trace if available
    trace = {
        "model": HAIKU_MODEL_ID,
        "metric": "composite_plan_score (judge)",
        "num_trials": n_trials,
        "num_candidates": args.num_candidates,
        "max_bootstrapped_demos": args.max_bootstrapped_demos,
        "train_size": len(train),
        "val_size": len(val),
        "elapsed_sec": round(elapsed_sec, 2),
        "best_score": getattr(compiled, "score", None),
    }
    # MIPRO stashes per-trial scores on compiled.candidate_programs in some
    # versions; try to capture if present.
    candidates = getattr(compiled, "candidate_programs", None)
    if candidates:
        trace["per_candidate_scores"] = [
            {
                "score": getattr(c, "score", None),
                "subscore": getattr(c, "subscore", None),
            }
            for c in candidates
        ]

    args.out_trace.parent.mkdir(parents=True, exist_ok=True)
    args.out_trace.write_text(json.dumps(trace, indent=2, default=str))
    print(f"Wrote MIPRO trace: {args.out_trace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
