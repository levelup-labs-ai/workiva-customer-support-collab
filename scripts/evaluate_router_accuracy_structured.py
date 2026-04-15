#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience only
    load_dotenv = None

from scripts.evaluate_router_accuracy import (
    DEFAULT_DATASET_PATH,
    build_report,
    load_dataset,
    maybe_write_json,
    print_report,
    select_cases,
)

PROMPT_ROUTER_BASELINE_STRUCTURED = (
    "You are a support routing classifier. "
    "Classify a user request into exactly one of these categories: "
    "review_workflow, permissions, billing, escalation."
)


class RouteDecision(BaseModel):
    category: Literal["review_workflow", "permissions", "billing", "escalation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline router accuracy using structured output."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional random sample size for faster smoke tests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed used with --limit.",
    )
    parser.add_argument(
        "--failure-limit",
        type=int,
        default=10,
        help="How many misclassified cases to print.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable report JSON file.",
    )
    return parser.parse_args()


def evaluate_cases(client, dataset: list[dict], prompt: str) -> list[dict]:
    results: list[dict] = []
    total = len(dataset)

    for index, case in enumerate(dataset, start=1):
        raw_response = ""
        fallback_reason = ""
        predicted = ""

        try:
            response = client.responses.parse(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": case["user_input"]},
                ],
                text_format=RouteDecision,
                temperature=0.3,
                max_output_tokens=40,
            )
            raw_response = response.output_text.strip()
            if response.output_parsed is not None:
                predicted = response.output_parsed.category
            else:
                fallback_reason = "Structured parse returned no output_parsed value."
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"

        results.append(
            {
                "scenario_id": case["scenario_id"],
                "user_input": case["user_input"],
                "expected_category": case["category"],
                "predicted_category": predicted,
                "exact_match": predicted == case["category"],
                "raw_response": raw_response,
                "fallback_reason": fallback_reason,
            }
        )

        if index == 1 or index % 10 == 0 or index == total:
            print(f"[{index}/{total}] evaluated {case['scenario_id']}", file=sys.stderr)

    return results


def main() -> int:
    args = parse_args()

    if load_dotenv is not None:
        load_dotenv()

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: openai. Run this with the repo venv, for example "
            "`.venv/bin/python scripts/evaluate_router_accuracy_structured.py`."
        ) from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    dataset_path = args.dataset.resolve()
    dataset = load_dataset(dataset_path)
    selected_dataset = select_cases(dataset, limit=args.limit, seed=args.seed)

    if not selected_dataset:
        raise ValueError("No cases selected for evaluation.")

    client = OpenAI()
    results = evaluate_cases(
        client=client,
        dataset=selected_dataset,
        prompt=PROMPT_ROUTER_BASELINE_STRUCTURED,
    )
    report = build_report(
        prompt_name="baseline_structured",
        prompt_text=PROMPT_ROUTER_BASELINE_STRUCTURED,
        dataset_path=dataset_path,
        results=results,
    )
    print_report(report, failure_limit=args.failure_limit)
    maybe_write_json(args.json_output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
