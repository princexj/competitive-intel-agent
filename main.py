"""Command-line entry point for SignalForge."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from graph import compare_targets, result_from_state, stream_target
from storage import save_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evidence-backed competitive intelligence."
    )
    parser.add_argument("targets", nargs="+", help="One to four research targets.")
    parser.add_argument("--groq-key", default=os.getenv("GROQ_API_KEY", ""))
    parser.add_argument("--tavily-key", default=os.getenv("TAVILY_API_KEY", ""))
    parser.add_argument("--output", default="competitive_intelligence.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.targets) > 4:
        raise SystemExit("Choose at most four targets.")

    groq_key = args.groq_key or getpass.getpass("Groq API key: ")
    tavily_key = args.tavily_key or getpass.getpass("Tavily API key: ")
    if not groq_key or not tavily_key:
        raise SystemExit("Groq and Tavily API keys are required.")

    results = []
    for target in args.targets:
        print(f"\nResearching {target}")
        state = {"topic": target, "errors": []}
        for node, update in stream_target(target, groq_key, tavily_key):
            print(f"  [{node}] complete")
            for key, value in update.items():
                if key == "errors":
                    state["errors"].extend(value)
                else:
                    state[key] = value
        results.append(result_from_state(target, state))

    comparison = compare_targets(results, groq_key)
    sections = []
    if comparison:
        sections.append("# Competitive Landscape\n\n" + comparison)
    sections.extend(result["report"] for result in results)
    Path(args.output).write_text("\n\n---\n\n".join(sections), encoding="utf-8")
    run_id = save_run(results, comparison)
    print(f"\nSaved {args.output} (run {run_id})")


if __name__ == "__main__":
    main()
