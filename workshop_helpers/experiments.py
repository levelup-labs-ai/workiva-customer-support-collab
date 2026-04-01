from datetime import datetime
import json

import pandas as pd
from arize import ArizeClient

from workshop_helpers.backend import TOOLS, hydrate_backend_from_dataset, run_support_agent_threadsafe
from workshop_helpers.metrics import build_evaluators, pack_response_payload

DATASET_NAME = "cs-support-workshop-benchmark"

VARIANT_BEHAVIORS = {
    "router": "Routing agent. It should read the customer message and return exactly one supported category label.",
    "v1": "Prompt-only assistant with no backend access. It should avoid bluffing, avoid invented actions, and ask one or more focused follow-ups when information is missing.",
    "v2": "Static-context assistant. It should use the provided support snapshot specifically, but it must not pretend a backend action already happened.",
    "v3": "Tool-using agent. When enough information exists, it should verify with tools and complete the backend action before replying.",
}


def dataset_index(dataset: list[dict]) -> dict:
    return {case["scenario_id"]: case for case in dataset}


def select_cases(dataset: list[dict], limit_n: int | None = None) -> list[dict]:
    if limit_n is None:
        return list(dataset)
    return list(dataset[:limit_n])


def summarize_dataset(dataset: list[dict]) -> dict:
    frame = pd.DataFrame(dataset)
    summary = {
        "scenario_count": len(frame),
        "categories": sorted(frame.category.unique()),
    }
    if "is_edge_case" in frame.columns:
        summary["standard_count"] = int(frame[~frame.is_edge_case].shape[0])
        summary["edge_case_count"] = int(frame[frame.is_edge_case].shape[0])
    return summary


def build_arize_dataframe(dataset: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": case["scenario_id"],
                "category": case["category"],
                "customer_id": case["source_data"].get("customer_id", ""),
                "user_input": case["user_input"],
                "expected_output": case["expected_output"],
            }
            for case in dataset
        ]
    )


def ensure_arize_dataset(arize_api_key: str, arize_space_id: str, dataset: list[dict]) -> dict:
    client = ArizeClient(api_key=arize_api_key)
    dataset_frame = build_arize_dataframe(dataset)
    list_response = client.datasets.list(space=arize_space_id)

    existing = next((item for item in list_response.datasets if item.name == DATASET_NAME), None)
    if existing:
        dataset_id = existing.id
        created = False
    else:
        response = client.datasets.create(
            space=arize_space_id,
            name=DATASET_NAME,
            examples=dataset_frame,
        )
        dataset_id = response.id
        created = True

    return {
        "client": client,
        "dataset_id": dataset_id,
        "dataset_name": DATASET_NAME,
        "row_count": len(dataset_frame),
        "created": created,
        "dataframe": dataset_frame,
    }


def build_tasks(
    client,
    dataset: list[dict],
    prompt_router: str,
    prompt_v1: str,
    prompt_v2: str,
    prompt_v3: str,
) -> dict:
    cases_by_id = dataset_index(dataset)
    categories = sorted({case["category"] for case in dataset})

    def task_router(row: dict) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_router.format(categories=", ".join(categories))},
                {"role": "user", "content": row["user_input"]},
            ],
            temperature=0,
            max_tokens=20,
        )
        return pack_response_payload(response.choices[0].message.content.strip())

    def task_v1(row: dict) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_v1},
                {"role": "user", "content": row["user_input"]},
            ],
            temperature=0.3,
            max_tokens=220,
        )
        return pack_response_payload(response.choices[0].message.content.strip())

    def task_v2(row: dict) -> str:
        case = cases_by_id.get(row["scenario_id"])
        if not case:
            return "Error: case not found"
        message = (
            f"Customer context (internal support snapshot):\n{json.dumps(case['source_data'], indent=2)}"
            f"\n\nCustomer message:\n{row['user_input']}"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_v2},
                {"role": "user", "content": message},
            ],
            temperature=0.3,
            max_tokens=220,
        )
        return pack_response_payload(response.choices[0].message.content.strip())

    def task_v3(row: dict) -> str:
        result = run_support_agent_threadsafe(
            customer_message=row["user_input"],
            instructions=prompt_v3.format(
                authenticated_customer_id=row.get("customer_id") or "UNKNOWN"
            ),
        )
        return pack_response_payload(
            result["output"],
            tool_calls=result.get("tool_calls", []),
            action_calls=result.get("action_calls", []),
        )

    return {"task_router": task_router, "task_v1": task_v1, "task_v2": task_v2, "task_v3": task_v3}


def run_experiment(arize_client, dataset_id: str, name_prefix: str, task, evaluators, concurrency: int = 3):
    experiment_name = f"{name_prefix}-{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    experiment, results_df = arize_client.experiments.run(
        name=experiment_name,
        dataset=dataset_id,
        task=task,
        evaluators=evaluators,
        concurrency=concurrency,
    )
    return {"experiment": experiment, "results_df": results_df, "experiment_name": experiment_name}


def production_readiness_checklist() -> list[tuple[str, str, str]]:
    return [
        ("correct_outcome improves from v1 to v3", "check experiment comparison", "agents should outperform prompt-only assistance on the same case set"),
        ("workflow_fit improves from v1 to v3", "check experiment comparison", "each variant should behave like its intended capability stage"),
        ("tone Good or Acceptable stays high", "check all three variants", "tone is a guardrail, not the main success metric"),
        ("human review gate before response sent to customer", "not yet designed", "required for v1: agent drafts, human approves"),
        ("tool-driven actions correspond to real downstream workflows", "stubbed in demo backend", "verify this before any real pilot"),
        ("sample size is appropriate for the workshop runtime", "configurable via LIMIT_N_CASES", "smaller runs are faster but noisier"),
        ("threshold to advance from demo to pilot", "not defined", "set based on stable outcome and workflow-fit performance"),
    ]


def format_checklist_rows(checklist: list[tuple[str, str, str]]) -> list[str]:
    rows = [f"{'#':<3} {'CRITERION':<44} {'STATUS':<34} NOTE", "-" * 118]
    for index, (criterion, status, note) in enumerate(checklist, start=1):
        icon = "OK" if any(word in status.lower() for word in ["check", "configurable", "stubbed"]) else "--"
        rows.append(f"{icon} {index:<2} {criterion:<44} {status:<34} {note}")
    return rows


def prepare_experiment_bundle(
    client,
    arize_api_key: str,
    arize_space_id: str,
    dataset: list[dict],
    prompt_router: str,
    prompt_v1: str,
    prompt_v2: str,
    prompt_v3: str,
    limit_n: int | None = None,
) -> dict:
    selected_dataset = select_cases(dataset, limit_n=limit_n)
    hydrate_backend_from_dataset(dataset)
    dataset_lookup = dataset_index(selected_dataset)
    arize_bundle = ensure_arize_dataset(arize_api_key, arize_space_id, selected_dataset)
    return {
        **arize_bundle,
        "dataset_lookup": dataset_lookup,
        "selected_dataset": selected_dataset,
        "tasks": build_tasks(client, selected_dataset, prompt_router, prompt_v1, prompt_v2, prompt_v3),
        "build_evaluators": lambda variant_name: build_evaluators(
            client,
            dataset_lookup,
            variant_name=variant_name,
            variant_behavior=VARIANT_BEHAVIORS[variant_name],
        ),
        "tool_count": len(TOOLS),
        "limit_n": limit_n,
    }
