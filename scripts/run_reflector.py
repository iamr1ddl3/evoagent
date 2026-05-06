#!/usr/bin/env python
"""CLI: run the Reflector against the last N days of CodeOrch errors.

Usage:
    python scripts/run_reflector.py [--since 7d] [--limit 100] [--out reports/reflector.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)

from agents.reflector import reflect  # noqa: E402


_DURATION_RE = re.compile(r"^(\d+)([smhdw])$")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def parse_duration(s: str) -> timedelta:
    m = _DURATION_RE.match(s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad duration: {s!r} (expect e.g. 7d, 12h)")
    return timedelta(**{_UNITS[m.group(2)]: int(m.group(1))})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--since", type=parse_duration, default=timedelta(days=7),
                   help="time window to scan, e.g. 24h, 7d, 2w (default 7d)")
    p.add_argument("--limit", type=int, default=100,
                   help="max error observations to ingest (default 100)")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "reflector.json",
                   help="where to write the JSON output")
    args = p.parse_args()

    print(f"Reflector: scanning Langfuse since={args.since} limit={args.limit} ...")
    proposals = reflect(since=args.since, max_observations=args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(proposals, indent=2, default=str))

    meta = proposals.get("_meta", {})
    print()
    print("=" * 60)
    print(f"Observations: {meta.get('n_observations', 0)}  Groups: {meta.get('n_groups', 0)}")
    print(f"Model: {meta.get('model')}  Latency: {meta.get('latency_ms')} ms")
    print(f"Tokens: {meta.get('usage')}")
    print("=" * 60)
    print()
    print(f"Summary: {proposals.get('summary', '(none)')}")
    print()
    for i, mode in enumerate(proposals.get("modes", []), 1):
        print(f"[{i}] {mode['agent']} :: {mode['failure_class']}  (n={mode['frequency']})")
        print(f"    diagnosis: {mode['diagnosis']}")
        edit = mode.get("proposed_edit", {})
        print(f"    proposal ({edit.get('target_section')}): {edit.get('new_text', '')[:160]}")
        print(f"    rationale: {edit.get('rationale')}")
        print()

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
