"""DSPy wrapper around CodeOrch's Planner.

Why DSPy: the Planner's job is the textbook DSPy use case — a structured
input (spec) maps to a structured output (plan with tasks + acceptance
criteria). Wrapping it as a `dspy.Module` with a typed `Signature` lets
MIPROv2 optimize the system prompt against an evaluation metric on Day 9.

This module deliberately exposes the SAME contract as
`codeorch.agents.planner.Planner.execute` so MIPRO before/after comparisons
are apples-to-apples:

    DSPyPlanner()(spec="...") -> {"tasks": [...], "acceptance_criteria": [...]}

Two choices worth flagging:

1. We use `dspy.Predict` (single-call) rather than `dspy.ChainOfThought`.
   The CodeOrch Planner doesn't ask Sonnet for explicit reasoning —
   adding CoT would change the comparison baseline for MIPRO. Day 9 can
   A/B Predict-vs-CoT separately.

2. The Signature mirrors the Planner's docstring exactly (1-5 tasks,
   sequential T1/T2/T3 ids, easy/medium/hard complexity). This is what
   MIPRO will preserve as ground truth while it tunes the instructions.
"""

from __future__ import annotations

import json
import os
from typing import Any

import dspy
from pydantic import BaseModel, Field, field_validator


# --- typed I/O ---------------------------------------------------------------


class PlanTask(BaseModel):
    id: str = Field(description="Sequential id: T1, T2, T3, ...")
    description: str = Field(description="Concrete what-to-build description")
    language: str = Field(description="python | typescript | javascript | ...")
    complexity: str = Field(description="easy | medium | hard")
    acceptance_criterion: str = Field(
        description="A single observable check that proves this task is done"
    )

    @field_validator("complexity")
    @classmethod
    def _check_complexity(cls, v: str) -> str:
        if v not in {"easy", "medium", "hard"}:
            raise ValueError(f"complexity must be easy|medium|hard, got {v!r}")
        return v


class Plan(BaseModel):
    tasks: list[PlanTask] = Field(
        description="1 to 5 tasks, each self-contained for parallel consumption"
    )
    acceptance_criteria: list[str] = Field(
        description="Run-level criteria — what proves the whole plan succeeded"
    )

    @field_validator("tasks")
    @classmethod
    def _check_count(cls, v: list[PlanTask]) -> list[PlanTask]:
        if not (1 <= len(v) <= 5):
            raise ValueError(f"plan must have 1-5 tasks, got {len(v)}")
        return v


# --- DSPy signature ----------------------------------------------------------


class SpecToPlan(dspy.Signature):
    """Decompose a natural-language code spec into a structured task list.

    Downstream agents (Coder, Tester) consume the tasks IN PARALLEL — each
    task must be self-contained and unambiguous. Use 1-5 tasks; more than
    5 means you are over-decomposing. complexity is easy (<20 lines),
    medium (20-80 lines), or hard (80+ lines or multi-file).
    """

    spec: str = dspy.InputField(desc="The natural-language code specification.")
    plan: Plan = dspy.OutputField(
        desc="A structured plan: 1-5 tasks plus run-level acceptance_criteria."
    )


# --- module ------------------------------------------------------------------


class DSPyPlanner(dspy.Module):
    """DSPy module wrapping the spec -> plan transformation.

    Use as:
        planner = DSPyPlanner()
        result = planner(spec="Write a Python function that ...")
        plan_dict = result.plan_dict   # the dict downstream consumers expect
    """

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(SpecToPlan)

    def forward(self, spec: str) -> dspy.Prediction:
        out = self.predict(spec=spec)

        # DSPy returns out.plan as a Plan instance (validated by Pydantic).
        # Expose a dict view so callers can drop straight into the CodeOrch
        # contract without knowing about Pydantic.
        plan: Plan = out.plan
        plan_dict: dict[str, Any] = plan.model_dump()
        return dspy.Prediction(plan=plan, plan_dict=plan_dict)


# --- LM configuration --------------------------------------------------------


# CodeOrch's Sonnet 4.6 ID. Kept in sync with codeorch/agents/models.py SONNET.
PLANNER_MODEL_ID = "anthropic/claude-sonnet-4-6"

# Day 9 — demoted target for MIPRO optimization. The story:
#   Haiku-baseline scores LOWER than Sonnet on the LLM-as-judge metric,
#   giving MIPRO real headroom. Successful optimization means Haiku-with-
#   optimized-prompt approaches Sonnet quality at ~5x lower cost. Mirrors
#   Day 7 RouterBench: "optimization compresses the cost-quality curve."
HAIKU_MODEL_ID = "anthropic/claude-haiku-4-5"


def configure_lm(*, model_id: str = PLANNER_MODEL_ID, max_tokens: int = 2048) -> dspy.LM:
    """Build and register a DSPy LM globally.

    Reads ANTHROPIC_API_KEY from env. Caching is left ON (default) so MIPRO
    optimization runs don't re-bill identical (signature, input) pairs.

    Args:
        model_id: defaults to Sonnet 4.6 (the Day 8 baseline). Pass
            HAIKU_MODEL_ID for the Day 9 demotion target.
        max_tokens: generation cap.
    """
    if "ANTHROPIC_API_KEY" not in os.environ or not os.environ["ANTHROPIC_API_KEY"]:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set — load_dotenv(override=True) before configuring DSPy."
        )

    lm = dspy.LM(model=model_id, max_tokens=max_tokens)
    dspy.configure(lm=lm)
    return lm


# --- baseline metric ---------------------------------------------------------


def plan_validity_score(example: dspy.Example, prediction: dspy.Prediction, trace: Any | None = None) -> float:
    """A baseline metric MIPRO can optimize against.

    Score = average of:
        - schema validity (1.0 if Pydantic accepted the output, else 0.0)
        - task-count sanity (1.0 if 1-5 tasks, else 0.0)
        - acceptance-criteria coverage (1.0 if at least one run-level
          criterion exists AND each task has its own non-empty
          acceptance_criterion, else 0.0)

    This is intentionally a *cheap* metric. Day 9 may layer an LLM-as-judge
    metric on top for richer signal; we want a deterministic baseline first
    so MIPRO has something stable to optimize.
    """
    plan = getattr(prediction, "plan", None)
    if plan is None or not isinstance(plan, Plan):
        return 0.0

    schema_ok = 1.0  # if we got here, Pydantic already validated

    count_ok = 1.0 if 1 <= len(plan.tasks) <= 5 else 0.0

    ac_ok = 1.0 if (
        plan.acceptance_criteria
        and all(t.acceptance_criterion.strip() for t in plan.tasks)
    ) else 0.0

    return (schema_ok + count_ok + ac_ok) / 3.0


__all__ = [
    "DSPyPlanner",
    "Plan",
    "PlanTask",
    "SpecToPlan",
    "PLANNER_MODEL_ID",
    "HAIKU_MODEL_ID",
    "configure_lm",
    "plan_validity_score",
]
