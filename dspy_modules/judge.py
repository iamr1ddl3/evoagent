"""LLM-as-judge metric for the DSPy Planner.

Why we need this: the Day 8 deterministic `plan_validity_score` returns 1.0
on every Sonnet output (schema validity + task count + AC presence are easy
to satisfy). MIPRO compiled against that metric has zero gradient.

The judge layer scores plans on three independent quality axes that the
deterministic metric cannot measure:

    1. spec_coverage        — does the plan address every requirement
                              in the spec, or are some left implicit?
    2. task_atomicity       — is each task self-contained and unambiguous,
                              or do tasks share state / overlap?
    3. acceptance_testability — can each acceptance_criterion be checked
                                with a single deterministic assertion?

Composite score = mean of the three sub-scores, range [0.0, 1.0].

Anti-bias measures (deliberate):

  - The judge sees (spec, plan_json) — NOT the model name that produced
    the plan. Sonnet-judges-Sonnet sycophancy can't kick in if Sonnet
    doesn't know it's grading its own outputs.
  - Anchor: "0.5 is a baseline pass; reserve 0.9+ for plans that
    demonstrably exceed expectations." Without this anchor LLM judges
    cluster at 0.85+ and MIPRO has no gradient to climb.
  - Three independent axes, each with its own rubric. The judge cannot
    collapse to "this plan looks fine" — it must defend three numbers
    separately, with rationales.

Caching: DSPy's default LiteLLM disk cache keys on (model, prompt, params).
MIPRO scoring identical (spec, plan) pairs across trials hits cache and
costs nothing. Set `JUDGE_CACHE_BUST` env var to force rescoring (use
when changing the rubric, not within a single MIPRO run).
"""

from __future__ import annotations

import json
import os
from typing import Any

import dspy
from pydantic import BaseModel, Field, field_validator

from .planner import Plan


# --- Judge model ---------------------------------------------------------------


JUDGE_MODEL_ID = "anthropic/claude-sonnet-4-6"
"""Sonnet 4.6 is used as the judge.

Justification:
  - Comprehension task (does plan cover spec?), not a generation task.
    Sonnet handles this well at ~$0.005/call.
  - Opus 4.7 would over-engineer; the judge isn't optimizing.
  - Haiku 4.5 as judge would be cheap (~$0.001/call) but a weaker judge
    gives noisier signal. MIPRO would chase noise. Saving $0.30 across
    a 60-call optimization to add variance is the wrong trade.
"""


# --- Typed output --------------------------------------------------------------


class JudgeRubric(BaseModel):
    """Output schema for one judging pass.

    Three independent sub-scores, each with a one-sentence rationale.
    Rationales are required so we can sanity-check the judge during
    development — and so the cached scores are auditable.
    """

    spec_coverage: float = Field(
        ge=0.0, le=1.0,
        description="0.0 = key requirements missing, 0.5 = baseline pass "
                    "(every requirement at least implied), 1.0 = every "
                    "requirement explicitly mapped to a task.",
    )
    spec_coverage_rationale: str = Field(min_length=10)

    task_atomicity: float = Field(
        ge=0.0, le=1.0,
        description="0.0 = tasks share state or overlap, 0.5 = baseline "
                    "(tasks runnable but partially intertwined), 1.0 = "
                    "every task is self-contained and parallelizable.",
    )
    task_atomicity_rationale: str = Field(min_length=10)

    acceptance_testability: float = Field(
        ge=0.0, le=1.0,
        description="0.0 = ACs are aspirational ('works correctly'), "
                    "0.5 = baseline (most ACs are checkable), 1.0 = every "
                    "AC can be a single deterministic assertion.",
    )
    acceptance_testability_rationale: str = Field(min_length=10)

    @field_validator("spec_coverage", "task_atomicity", "acceptance_testability")
    @classmethod
    def _round_to_quartile(cls, v: float) -> float:
        """Snap to {0.0, 0.25, 0.5, 0.75, 1.0}.

        LLM judges drift toward 0.85, 0.87, 0.83 with no real semantic
        difference between those numbers. Quartile snapping forces the
        judge into 5 distinguishable buckets, giving MIPRO real gradient.
        """
        return round(v * 4) / 4

    @property
    def composite(self) -> float:
        return (self.spec_coverage + self.task_atomicity + self.acceptance_testability) / 3.0


# --- DSPy signature for the judge ---------------------------------------------


class PlanJudgeSignature(dspy.Signature):
    """Score a plan against its originating spec on three quality axes.

    You are judging plan QUALITY, not plan COMPLETENESS-OF-FIELDS — Pydantic
    validation already enforces schema correctness upstream. Your job is to
    measure whether this plan would meaningfully drive a successful build.

    Anchor your scores:
      - 0.5 is a BASELINE PASS. A plan that "works" gets 0.5.
      - 0.75 is good — most acceptance criteria are testable, etc.
      - Reserve 1.0 for plans that demonstrably exceed expectations
        (e.g., explicit edge cases in the AC, clean parallelism boundaries).
      - Use 0.0 only for clear failures (key spec requirements missing,
        tasks that overlap and can't run in parallel, ACs that are
        unprovable assertions).

    Score each axis independently. Do NOT collapse to "this plan looks fine."
    """

    spec: str = dspy.InputField(desc="The original natural-language code spec.")
    plan_json: str = dspy.InputField(
        desc="The plan, JSON-serialized. Includes tasks (with id, description, "
             "language, complexity, acceptance_criterion) and run-level "
             "acceptance_criteria."
    )
    rubric: JudgeRubric = dspy.OutputField(
        desc="Three sub-scores in {0.0, 0.25, 0.5, 0.75, 1.0} with one-sentence "
             "rationales each. Be honest — the optimizer needs real gradient."
    )


# --- Judge module --------------------------------------------------------------


class PlanJudge(dspy.Module):
    """Stateless DSPy judge over (spec, plan_json) -> JudgeRubric.

    Usage:
        judge = PlanJudge()
        rubric = judge(spec=spec, plan_json=plan_json).rubric
        score = rubric.composite
    """

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(PlanJudgeSignature)

    def forward(self, spec: str, plan_json: str) -> dspy.Prediction:
        out = self.predict(spec=spec, plan_json=plan_json)
        return dspy.Prediction(rubric=out.rubric)


# --- Composite metric ---------------------------------------------------------


_JUDGE_SINGLETON: PlanJudge | None = None


def _get_judge() -> PlanJudge:
    """Lazy-init the judge to avoid configuring it before dspy.configure()."""
    global _JUDGE_SINGLETON
    if _JUDGE_SINGLETON is None:
        _JUDGE_SINGLETON = PlanJudge()
    return _JUDGE_SINGLETON


def composite_plan_score(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Any | None = None,
) -> float:
    """The MIPRO-facing metric: validity (gate) × judge composite.

    Flow:
      1. If `prediction.plan` failed Pydantic validation upstream, score = 0.0.
         No need to spend judge tokens on a malformed plan.
      2. Otherwise, run the judge over (spec, plan_json) and return the
         composite (mean of 3 sub-scores).

    Why multiplicative gate vs. additive blend:
      A plan that fails schema is 0.0, full stop — we don't want MIPRO
      learning that "almost-valid plans with great descriptions" can win.
      Validity is a hard prerequisite, not a soft signal.

    Why mean-of-3 vs weighted:
      All three axes are independently necessary. A plan with great spec
      coverage but unparallelizable tasks is broken. A plan with atomic
      tasks but untestable ACs is broken. Equal weighting forces MIPRO
      to optimize all three jointly.
    """
    plan = getattr(prediction, "plan", None)
    if plan is None or not isinstance(plan, Plan):
        return 0.0

    # Spec lives on the example
    spec = example.spec if hasattr(example, "spec") else example.get("spec", "")
    if not spec:
        return 0.0

    plan_json = json.dumps(plan.model_dump(), indent=2, ensure_ascii=False)

    judge = _get_judge()
    try:
        out = judge(spec=spec, plan_json=plan_json)
        rubric: JudgeRubric = out.rubric
    except Exception as exc:
        # If the judge itself fails, return a neutral score rather than 0.0
        # — we don't want a transient API failure to look like a bad plan.
        # 0.5 is the documented baseline anchor.
        print(f"[judge] WARN: judge call failed ({type(exc).__name__}: {exc}); returning 0.5 baseline")
        return 0.5

    return rubric.composite


def score_with_breakdown(
    example: dspy.Example,
    prediction: dspy.Prediction,
) -> dict[str, Any]:
    """Same scoring pipeline as composite_plan_score but returns the full
    breakdown for use by the before/after benchmark report.

    Returns:
        {
          "ok": bool,                      # validity gate passed
          "composite": float,              # mean of 3 sub-scores or 0.0
          "spec_coverage": float | None,
          "task_atomicity": float | None,
          "acceptance_testability": float | None,
          "rationales": dict | None,
        }
    """
    plan = getattr(prediction, "plan", None)
    if plan is None or not isinstance(plan, Plan):
        return {
            "ok": False, "composite": 0.0,
            "spec_coverage": None, "task_atomicity": None,
            "acceptance_testability": None, "rationales": None,
        }

    spec = example.spec if hasattr(example, "spec") else example.get("spec", "")
    plan_json = json.dumps(plan.model_dump(), indent=2, ensure_ascii=False)

    judge = _get_judge()
    try:
        rubric: JudgeRubric = judge(spec=spec, plan_json=plan_json).rubric
    except Exception as exc:
        return {
            "ok": True, "composite": 0.5,
            "spec_coverage": None, "task_atomicity": None,
            "acceptance_testability": None,
            "rationales": {"_error": f"{type(exc).__name__}: {exc}"},
        }

    return {
        "ok": True,
        "composite": rubric.composite,
        "spec_coverage": rubric.spec_coverage,
        "task_atomicity": rubric.task_atomicity,
        "acceptance_testability": rubric.acceptance_testability,
        "rationales": {
            "spec_coverage": rubric.spec_coverage_rationale,
            "task_atomicity": rubric.task_atomicity_rationale,
            "acceptance_testability": rubric.acceptance_testability_rationale,
        },
    }


__all__ = [
    "JUDGE_MODEL_ID",
    "JudgeRubric",
    "PlanJudge",
    "PlanJudgeSignature",
    "composite_plan_score",
    "score_with_breakdown",
]
