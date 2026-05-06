"""Reflector agent — reads CodeOrch failure traces from Langfuse, classifies
failure modes, and emits structured prompt-edit proposals.

The Reflector is the "left lung" of EvoAgent. It produces the corrective
signal that DSPy/MIPRO will turn into a compiled prompt on Day 9.

Pipeline:
    1. Pull error-level observations from Langfuse (last N days, codeorch tag).
    2. Group by (agent_name, error_class) — the natural failure-mode key.
    3. Sample one representative trace per group (full input + status_message)
       so the LLM has concrete evidence, not a count alone.
    4. Ask Sonnet 4.6 to classify failure modes and propose edits to the
       agent's system prompt that would have prevented each one.
    5. Return a list of FailureModeProposal objects, each grounded in the
       traces it cites.

Output schema (FailureModeProposals):
    {
      "modes": [
        {
          "agent": "Planner|Coder|Tester|QualityGate|Reviewer|Documenter",
          "failure_class": "schema-violation|tool-misuse|under-decomposition|...",
          "frequency": int,                       # number of traces in group
          "evidence_trace_ids": [str, ...],
          "diagnosis": str,                       # what went wrong, why
          "proposed_edit": {
            "target_section": "rules|format|examples",
            "new_text": str,                      # proposed insertion / replacement
            "rationale": str
          }
        }
      ],
      "summary": str
    }

Why Langfuse-as-source (not pgvector): the agent_context table records
Python-level failures (status='failure', error string captured). Langfuse
records *any* observation tagged level=ERROR — including LLM-side issues
that didn't crash the agent (degraded outputs, retry-then-recover paths,
upstream span errors). The richer signal is on Langfuse.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

# Reuse codeorch's Anthropic client + routing table without copying. We
# import the codeorch package under an aliased namespace to dodge the
# `agents.*` name collision (evoagent has its own `agents/` package).
import importlib.util
import types

_CODEORCH_AGENTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "codeorch", "agents")
)


def _load_codeorch_module(filename: str, alias: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        alias, os.path.join(_CODEORCH_AGENTS_DIR, filename)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load codeorch module {filename}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_codeorch_client = _load_codeorch_module("anthropic_client.py", "_codeorch_anthropic_client")
_codeorch_models = _load_codeorch_module("models.py", "_codeorch_models")
call_model = _codeorch_client.call_model
ROUTING = _codeorch_models.ROUTING

from langfuse.api import LangfuseAPI  # noqa: E402


REFLECTOR_MODEL = ROUTING["planner"]["model"]  # Sonnet 4.6 — same class as the
# agents being analyzed. Using a smarter model than the
# target risks hallucinating failure modes the target
# couldn't actually fix.

REFLECTOR_SYSTEM = """You are the Reflector — a meta-agent that diagnoses
failures in a multi-agent code-generation pipeline (CodeOrch) and proposes
prompt edits.

You will receive a structured failure report grouped by (agent, error_class).
For each group, propose ONE concrete edit to that agent's system prompt that
would prevent the failure mode from recurring.

Output ONLY valid JSON matching this schema, no prose, no markdown fences:

{
  "modes": [
    {
      "agent": "<one of: Planner | Coder | Tester | QualityGate | Reviewer | Documenter>",
      "failure_class": "<short kebab-case label: e.g. schema-violation, json-format, under-decomposition, missing-edge-case>",
      "frequency": <int>,
      "evidence_trace_ids": ["<trace_id>", ...],
      "diagnosis": "<2-3 sentences: what the agent produced, why it was wrong>",
      "proposed_edit": {
        "target_section": "<rules | format | examples>",
        "new_text": "<the actual text to insert into the system prompt>",
        "rationale": "<one sentence: why this edit fixes the failure class>"
      }
    }
  ],
  "summary": "<2-3 sentences over all modes>"
}

Rules:
- proposed_edit.new_text must be drop-in usable — write it as if it will be
  pasted directly into the agent's system prompt.
- One mode per (agent, failure_class) group; do not split.
- frequency must equal the count of traces shown for that group.
- Do not invent failure classes that aren't represented in the input.
"""


@dataclass
class ErrorObservation:
    trace_id: str
    observation_id: str
    agent_name: str
    error_class: str
    status_message: str
    started_at: datetime
    tags: list[str]


def _parse_agent_and_class(name: str | None, status_message: str | None) -> tuple[str, str]:
    """Extract (agent, error_class) keys from a Langfuse observation.

    name examples:
        agent.Planner.llm
        agent.QualityGate.sonnet-verdict
        codeorch.run

    error_class is the type-name prefix of status_message:
        'ValueError: model returned non-JSON: ...'   -> 'ValueError'
        'AgentFailure: model returned non-JSON: ...' -> 'AgentFailure'
        None                                          -> 'unknown'
    """
    agent = "unknown"
    if name:
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "agent":
            agent = parts[1]
        elif name == "codeorch.run":
            agent = "Orchestrator"

    error_class = "unknown"
    if status_message:
        head = status_message.split(":", 1)[0].strip()
        # Heuristic: error_class is one CamelCase token (Python exception name)
        if head and head.replace("_", "").isalnum() and head[0].isupper():
            error_class = head

    return agent, error_class


def _client() -> LangfuseAPI:
    return LangfuseAPI(
        base_url=os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        username=os.environ["LANGFUSE_PUBLIC_KEY"],
        password=os.environ["LANGFUSE_SECRET_KEY"],
    )


def fetch_error_observations(
    *,
    since: timedelta = timedelta(days=7),
    limit: int = 100,
    project_tag: str = "codeorch",
) -> list[ErrorObservation]:
    """Query Langfuse for error-level observations across CodeOrch runs.

    Trace.list lets us filter by tag; observations.get_many lets us filter
    by level. We do both: scope to traces that carry the codeorch tag, then
    walk those traces for ERROR observations with name + status_message.
    """
    client = _client()
    cutoff = datetime.now(timezone.utc) - since

    # Step 1: list candidate traces (scoped by tag + time window).
    candidate_trace_ids: set[str] = set()
    page = 1
    while True:
        traces = client.trace.list(
            tags=[project_tag],
            from_timestamp=cutoff,
            limit=50,
            page=page,
        )
        if not traces.data:
            break
        candidate_trace_ids.update(t.id for t in traces.data)
        if len(traces.data) < 50:
            break
        page += 1
        if page > 20:  # safety
            break

    # Step 2: fetch each trace with full details and harvest ERROR obs.
    out: list[ErrorObservation] = []
    for tid in candidate_trace_ids:
        try:
            trace = client.trace.get(tid)
        except Exception:
            continue
        for obs in trace.observations or []:
            if obs.level != "ERROR":
                continue
            agent, error_class = _parse_agent_and_class(obs.name, obs.status_message)
            out.append(
                ErrorObservation(
                    trace_id=tid,
                    observation_id=obs.id,
                    agent_name=agent,
                    error_class=error_class,
                    status_message=obs.status_message or "",
                    started_at=obs.start_time,
                    tags=trace.tags or [],
                )
            )
            if len(out) >= limit:
                return out
    return out


def group_by_failure_mode(
    obs_list: list[ErrorObservation],
) -> dict[tuple[str, str], list[ErrorObservation]]:
    """Group observations by (agent, error_class). Skips the SPAN-level
    AgentFailure wrapper when a more specific GENERATION-level error exists
    in the same trace — the inner error is the actual cause."""
    groups: dict[tuple[str, str], list[ErrorObservation]] = defaultdict(list)
    for o in obs_list:
        groups[(o.agent_name, o.error_class)].append(o)
    # Order groups by frequency desc so the LLM sees the worst offender first.
    return dict(
        sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    )


def _render_input_block(
    groups: dict[tuple[str, str], list[ErrorObservation]],
    sample_per_group: int = 2,
) -> str:
    lines = []
    for (agent, error_class), items in groups.items():
        lines.append(f"## Agent={agent} | error_class={error_class} | count={len(items)}")
        for ex in items[:sample_per_group]:
            msg = ex.status_message[:600].replace("\n", " ")
            lines.append(f"  - trace_id={ex.trace_id} status_message={msg!r}")
        lines.append("")
    return "\n".join(lines)


def reflect(
    *,
    since: timedelta = timedelta(days=7),
    max_observations: int = 100,
) -> dict[str, Any]:
    """Top-level Reflector entrypoint.

    Returns the parsed JSON output of the LLM call (FailureModeProposals shape).
    Raises if Langfuse fetch returns no errors — there's nothing to reflect on.
    """
    obs = fetch_error_observations(since=since, limit=max_observations)
    if not obs:
        return {"modes": [], "summary": "No error-level observations found in the time window."}

    groups = group_by_failure_mode(obs)
    user_block = _render_input_block(groups)

    call = call_model(
        model=REFLECTOR_MODEL,
        system=REFLECTOR_SYSTEM,
        user=f"Failure observations grouped by (agent, error_class):\n\n{user_block}",
        max_tokens=2048,
    )

    proposals = call.parsed or {}
    if not isinstance(proposals.get("modes"), list):
        raise ValueError(f"Reflector returned malformed output: {call.text[:300]!r}")

    # Attach metadata so callers can audit the run without rerunning the LLM.
    proposals["_meta"] = {
        "n_observations": len(obs),
        "n_groups": len(groups),
        "model": call.model,
        "usage": call.usage,
        "latency_ms": call.latency_ms,
        "since_seconds": int(since.total_seconds()),
    }
    return proposals
