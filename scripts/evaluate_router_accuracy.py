#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience only
    load_dotenv = None

from workshop_helpers.scenarios import run_router_raw

DEFAULT_DATASET_PATH = REPO_ROOT / "workshop_helpers" / "dataset.json"

PROMPT_ROUTER_BASELINE = (
    "You are a support routing classifier. "
    "Classify a user request into one of the 4 categories: review_workflow, permissions, billing, escalation. "
    "Return ONLY a JSON object with the key 'category' and the appropriate category value."
)

PROMPT_ROUTER_REFINED = """You are a support routing classifier. Read the customer message and return exactly one category as raw JSON.

Categories:
- review_workflow: The customer is asking about workflow state or process, such as which review step a file is in, what is blocking progress, who still needs to approve, whether a reviewer or approver is required, whether a step can be skipped, what happens next, why the file moved backward, why a workflow exception happened, how to unblock a workflow, or why submit/publish is unavailable.
- permissions: The customer is asking about access rights, sharing, authorization, request path, who can grant access, why access was denied, why an access request is stuck, how to restore access, temporary access, cross-workspace access, contractor access, or how to bypass or speed up the access request path.
- billing: The customer is asking for billing operations help, such as explanation of an invoice, charge, refund, credit, fee, tax, proration, quote mismatch, payment issue, or what a billing response should say.
- escalation: The customer needs human ownership because either the issue is severe or it falls outside the supported workflow/permissions/billing taxonomy. This includes security or privacy exposure, legal or contract risk, abusive behavior, repeated severe unresolved failures, outage-like impact, explicit requests for a human or senior person, major deadline risk, product defects or technical investigation, data recovery or restore requests, sync/import/export/search/notification/integration failures, and commercial or account-management conversations such as competitor comparisons, discount requests, renewal negotiation, pricing strategy, procurement discussion, or contract-option review.

Tie-breakers:
1. Choose the customer's primary ask, not every topic mentioned.
2. If the message asks how to get access, who can grant access, why access was denied, or why an access request is pending or blocked, choose permissions even if workflow language is also present, unless the message says the access problem is causing severe business risk that needs immediate human attention, such as threatening a board meeting, filing, or comparable critical deadline.
3. If the message asks about workflow stage, approvers, blockers, required review steps, exceptions, publish/submit behavior, or how to unblock the process, choose review_workflow even if access is part of the surrounding context.
4. When access loss is mentioned only as one reason a workflow is stalled, and the customer is asking how to move the workflow forward, choose review_workflow.
5. Choose billing only for billing operations questions about explaining or handling a bill, invoice, charge, credit, refund, tax, fee, payment detail, or quote mismatch. Invoice-vs-quote questions stay in billing even if executives, urgency, or dissatisfaction are mentioned.
6. If pricing or invoice language is present but the real ask is competitor comparison, discount negotiation, commercial options, vendor evaluation, procurement, renewal strategy, or account-management discussion, choose escalation, not billing.
7. If the request is mainly about diagnosing, investigating, restoring, recovering, or escalating a product or platform problem rather than about workflow state, permissions, or billing operations, choose escalation.
8. Ordinary urgency alone does not force escalation, but severe business risk, explicit human-handoff requests, or unsupported issue types do. This includes access problems that put a board meeting, filing, or similarly critical event at immediate risk.

Return only this JSON shape: {"category": "review_workflow"}
Do not include explanations, markdown, or code fences.
"""

PROMPT_VARIANTS = {
    "baseline": PROMPT_ROUTER_BASELINE,
    "refined": PROMPT_ROUTER_REFINED,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate router prompt accuracy against a dataset.json file."
    )
    parser.add_argument(
        "--prompt",
        choices=sorted(PROMPT_VARIANTS),
        default="baseline",
        help="Prompt variant to run. Defaults to the V1 baseline prompt from the notebook.",
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


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    data = json.loads(dataset_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of cases in {dataset_path}.")
    return data


def select_cases(dataset: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None:
        return list(dataset)
    sampled = list(dataset)
    random.Random(seed).shuffle(sampled)
    return sampled[:limit]


def evaluate_cases(
    client: Any,
    dataset: list[dict[str, Any]],
    prompt: str,
    categories: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(dataset)

    for index, case in enumerate(dataset, start=1):
        route = run_router_raw(client, case["user_input"], prompt, categories)
        predicted = route.get("category", "")
        result = {
            "scenario_id": case["scenario_id"],
            "user_input": case["user_input"],
            "expected_category": case["category"],
            "predicted_category": predicted,
            "exact_match": predicted == case["category"],
            "raw_response": route.get("raw_response", ""),
            "fallback_reason": route.get("fallback_reason", ""),
        }
        results.append(result)

        if index == 1 or index % 10 == 0 or index == total:
            print(f"[{index}/{total}] evaluated {case['scenario_id']}", file=sys.stderr)

    return results


def build_report(
    prompt_name: str,
    prompt_text: str,
    dataset_path: Path,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for row in results if row["exact_match"])
    invalid_outputs = sum(1 for row in results if row["fallback_reason"])

    by_expected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        by_expected[row["expected_category"]].append(row)
        confusion[row["expected_category"]][row["predicted_category"]] += 1

    per_category = []
    for category in sorted(by_expected):
        rows = by_expected[category]
        category_correct = sum(1 for row in rows if row["exact_match"])
        per_category.append(
            {
                "category": category,
                "rows_evaluated": len(rows),
                "correct": category_correct,
                "accuracy": category_correct / len(rows) if rows else None,
            }
        )

    return {
        "prompt_variant": prompt_name,
        "prompt_text": prompt_text,
        "dataset_path": str(dataset_path.resolve()),
        "rows_evaluated": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "invalid_category_outputs": invalid_outputs,
        "per_category_accuracy": per_category,
        "confusion_matrix": {
            expected: dict(sorted(predicted.items()))
            for expected, predicted in sorted(confusion.items())
        },
        "misclassified_cases": [row for row in results if not row["exact_match"]],
        "results": results,
    }


def print_report(report: dict[str, Any], failure_limit: int) -> None:
    accuracy = report["accuracy"]
    accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "n/a"

    print(f"Prompt variant: {report['prompt_variant']}")
    print(f"Dataset: {report['dataset_path']}")
    print(f"Rows evaluated: {report['rows_evaluated']}")
    print(f"Overall accuracy: {accuracy_text} ({report['correct']}/{report['rows_evaluated']})")
    print(f"Invalid category outputs: {report['invalid_category_outputs']}")
    print()
    print("Per-category accuracy:")
    for row in report["per_category_accuracy"]:
        accuracy_text = f"{row['accuracy']:.2%}" if row["accuracy"] is not None else "n/a"
        print(f"- {row['category']}: {accuracy_text} ({row['correct']}/{row['rows_evaluated']})")

    failures = report["misclassified_cases"]
    print()
    print(f"Misclassified cases: {len(failures)}")
    for row in failures[:failure_limit]:
        print(
            f"- {row['scenario_id']}: expected={row['expected_category']} predicted={row['predicted_category']}"
        )
        print(f"  user_input: {row['user_input']}")
        if row["fallback_reason"]:
            print(f"  parse_note: {row['fallback_reason']}")
        print(f"  raw_response: {row['raw_response']}")

    if len(failures) > failure_limit:
        remaining = len(failures) - failure_limit
        print(f"... {remaining} more misclassified cases not shown")


def maybe_write_json(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print()
    print(f"Wrote JSON report to {path.resolve()}")


def main() -> int:
    args = parse_args()

    if load_dotenv is not None:
        load_dotenv()

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: openai. Run this with the repo venv, for example "
            "`.venv/bin/python scripts/evaluate_router_accuracy.py --prompt baseline`."
        ) from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    dataset_path = args.dataset.resolve()
    dataset = load_dataset(dataset_path)
    selected_dataset = select_cases(dataset, limit=args.limit, seed=args.seed)
    categories = sorted({case["category"] for case in dataset})

    if not selected_dataset:
        raise ValueError("No cases selected for evaluation.")

    client = OpenAI()
    prompt_text = PROMPT_VARIANTS[args.prompt]
    results = evaluate_cases(client, selected_dataset, prompt_text, categories)
    report = build_report(
        prompt_name=args.prompt,
        prompt_text=prompt_text,
        dataset_path=dataset_path,
        results=results,
    )

    print_report(report, failure_limit=args.failure_limit)
    maybe_write_json(args.json_output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
