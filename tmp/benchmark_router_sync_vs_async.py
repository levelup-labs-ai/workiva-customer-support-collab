import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from arize import ArizeClient
from arize.experiments.evaluators.base import EvaluationResult, Evaluator
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic import create_model

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workshop_helpers.data import DATASET
from workshop_helpers.experiments import build_arize_dataframe
from workshop_helpers.metrics import LABEL_SCORE, RoutingAccuracyEvaluator, pack_response_payload
from workshop_helpers.scenarios import run_router_structured
load_dotenv(".env")

PROMPT_ROUTER = """You are a support routing classifier. Your task is to classify customer support queries into one of the following categories and return ONLY a JSON response.

**Valid Categories:**
- review_workflow: Questions about file status, approval workflows, review steps, pending actions, or workflow exceptions
- permissions: Questions about access rights, sharing permissions, role-based access, or authorization
- billing: Questions about invoices, charges, pricing, or payment issues
- escalation: Urgent issues involving security breaches, sensitive data exposure, or critical problems requiring immediate human attention

**Instructions:**
1. Analyze the user's input to determine which category best fits their query
2. Return ONLY a JSON object with the key 'category' and the appropriate category value
3. Do NOT provide explanations or additional text in your response

**Output Format:**
{"category": "review_workflow"}

**Examples:**
- "How do I know if the file is waiting on me?" -> {"category": "review_workflow"}
- "Can contractors access this document?" -> {"category": "permissions"}
- "Why is my invoice higher this month?" -> {"category": "billing"}
- "We shared sensitive data with the wrong party" -> {"category": "escalation"}
"""

ROUTING_CATEGORIES = sorted({case["category"] for case in DATASET})
ROW_COUNT = len(DATASET)
CONCURRENCY = 10
ROUTER_DECISION_MODEL = create_model(
    "RouterDecision",
    category=(Literal.__getitem__(tuple(ROUTING_CATEGORIES)), ...),
)


class AsyncRoutingAccuracyEvaluator(Evaluator):
    async def async_evaluate(self, dataset_row, input, output, **kwargs):
        payload = json.loads(output or "{}")
        actual = payload.get("response_text", "")
        expected = dataset_row.get("category", "")
        exact = actual.strip().lower() == expected.strip().lower()
        label = "Good" if exact else "Poor"
        reasoning = (
            f"Exact match on normalized label `{expected}`."
            if exact
            else f"Predicted `{actual}` but expected `{expected}`."
        )
        return EvaluationResult(
            score=LABEL_SCORE[label],
            label=label,
            explanation=reasoning,
        )


def make_sync_task(client):
    def task(dataset_row):
        route = run_router_structured(
            client,
            dataset_row["user_input"],
            PROMPT_ROUTER,
            ROUTING_CATEGORIES,
        )
        return pack_response_payload(route["category"], metadata=route)

    return task


def make_async_task(client):
    async def task(dataset_row):
        response = await client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": PROMPT_ROUTER},
                {"role": "user", "content": dataset_row["user_input"]},
            ],
            temperature=0,
            max_output_tokens=80,
            text_format=ROUTER_DECISION_MODEL,
        )
        parsed = response.output_parsed
        if parsed is None:
            category = response.output_text.strip()
            route = {"category": category}
        elif hasattr(parsed, "model_dump"):
            route = parsed.model_dump()
        else:
            route = dict(parsed)
        if route.get("category") not in ROUTING_CATEGORIES:
            route["fallback_reason"] = f"Invalid category returned: {route.get('category', '')}"
            route["category"] = "escalation"
        return pack_response_payload(route["category"], metadata=route)

    return task


def ensure_benchmark_dataset(arize_client, space_id):
    dataset_name = f"workiva-router-benchmark-50-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    dataset_frame = build_arize_dataframe(DATASET)
    response = arize_client.datasets.create(
        space=space_id,
        name=dataset_name,
        examples=dataset_frame,
    )
    return dataset_name, response.id


def run_case(arize_client, dataset_id, task, evaluators):
    started = time.perf_counter()
    experiment, results_df = arize_client.experiments.run(
        name=f"router-benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        dataset=dataset_id,
        task=task,
        evaluators=evaluators,
        concurrency=CONCURRENCY,
        exit_on_error=True,
        timeout=180,
    )
    elapsed = time.perf_counter() - started
    score_col = next(col for col in results_df.columns if col.endswith(".score"))
    return {
        "experiment": experiment,
        "results_df": results_df,
        "elapsed_seconds": elapsed,
        "rows": len(results_df),
        "accuracy": float(results_df[score_col].mean()),
    }


def main():
    sync_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    async_client = AsyncOpenAI()
    arize_space_id = os.environ["ARIZE_SPACE_ID"]
    arize_api_key = os.environ["ARIZE_API_KEY"]
    arize_client = ArizeClient(api_key=arize_api_key)

    dataset_name, dataset_id = ensure_benchmark_dataset(arize_client, arize_space_id)

    sync_result = run_case(
        arize_client,
        dataset_id,
        make_sync_task(sync_client),
        [RoutingAccuracyEvaluator(expected_field="category")],
    )
    async_result = run_case(
        arize_client,
        dataset_id,
        make_async_task(async_client),
        [AsyncRoutingAccuracyEvaluator()],
    )

    summary = {
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "row_count": ROW_COUNT,
        "concurrency": CONCURRENCY,
        "sync_seconds": round(sync_result["elapsed_seconds"], 2),
        "async_seconds": round(async_result["elapsed_seconds"], 2),
        "delta_seconds": round(sync_result["elapsed_seconds"] - async_result["elapsed_seconds"], 2),
        "delta_percent": round(
            ((sync_result["elapsed_seconds"] - async_result["elapsed_seconds"]) / sync_result["elapsed_seconds"]) * 100,
            1,
        ),
        "sync_accuracy": sync_result["accuracy"],
        "async_accuracy": async_result["accuracy"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
