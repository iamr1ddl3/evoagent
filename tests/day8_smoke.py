"""Day 8 smoke test — Reflector + DSPy Planner contracts.

What this checks (cheap, no LLM calls):
  1. Reflector module imports + exposes the expected entrypoint
  2. _parse_agent_and_class extracts (agent, error_class) from observation names
  3. group_by_failure_mode buckets correctly and orders by frequency desc
  4. DSPyPlanner module + Plan/PlanTask schemas round-trip via Pydantic
  5. plan_validity_score returns 0.0 when prediction has no plan, 1.0 on a
     well-formed Plan
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Stub the API key so importing reflector / dspy_modules.planner doesn't crash
# if the dev hasn't sourced .env yet.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "x")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "x")


def test_reflector_imports():
    from agents import reflector  # noqa: F401
    from agents.reflector import (  # noqa: F401
        ErrorObservation,
        _parse_agent_and_class,
        group_by_failure_mode,
        reflect,
    )


def test_parse_agent_and_class():
    from agents.reflector import _parse_agent_and_class

    assert _parse_agent_and_class("agent.Planner.llm", "ValueError: bad json") == ("Planner", "ValueError")
    assert _parse_agent_and_class("agent.QualityGate.sonnet-verdict", "AgentFailure: x") == ("QualityGate", "AgentFailure")
    assert _parse_agent_and_class("codeorch.run", "RuntimeError: boom") == ("Orchestrator", "RuntimeError")
    assert _parse_agent_and_class(None, None) == ("unknown", "unknown")


def test_group_by_failure_mode_order():
    from agents.reflector import ErrorObservation, group_by_failure_mode

    obs = [
        ErrorObservation("t1", "o1", "Planner", "ValueError", "x", datetime.now(timezone.utc), []),
        ErrorObservation("t1", "o2", "Planner", "ValueError", "x", datetime.now(timezone.utc), []),
        ErrorObservation("t2", "o3", "QualityGate", "JSONError", "y", datetime.now(timezone.utc), []),
    ]
    groups = group_by_failure_mode(obs)
    keys = list(groups.keys())
    assert keys[0] == ("Planner", "ValueError"), "highest-frequency group should sort first"
    assert len(groups[("Planner", "ValueError")]) == 2


def test_dspy_planner_schema():
    from dspy_modules.planner import Plan, PlanTask

    plan = Plan(
        tasks=[
            PlanTask(
                id="T1",
                description="add two numbers",
                language="python",
                complexity="easy",
                acceptance_criterion="add(2,3) == 5",
            )
        ],
        acceptance_criteria=["function returns correct sum"],
    )
    d = plan.model_dump()
    assert d["tasks"][0]["complexity"] == "easy"
    assert len(d["acceptance_criteria"]) == 1


def test_plan_validity_score_paths():
    import dspy
    from dspy_modules.planner import Plan, PlanTask, plan_validity_score

    # No plan -> 0.0
    pred_empty = dspy.Prediction(plan=None)
    assert plan_validity_score(dspy.Example(spec="x"), pred_empty) == 0.0

    # Well-formed plan -> 1.0
    plan = Plan(
        tasks=[PlanTask(
            id="T1", description="d", language="python", complexity="easy",
            acceptance_criterion="ac",
        )],
        acceptance_criteria=["ok"],
    )
    pred_ok = dspy.Prediction(plan=plan)
    assert plan_validity_score(dspy.Example(spec="x"), pred_ok) == 1.0


if __name__ == "__main__":
    test_reflector_imports()
    test_parse_agent_and_class()
    test_group_by_failure_mode_order()
    test_dspy_planner_schema()
    test_plan_validity_score_paths()
    print("Day 8 smoke: 5/5 passed")
