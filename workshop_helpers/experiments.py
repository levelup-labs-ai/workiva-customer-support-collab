from datetime import datetime
import json
import random

import pandas as pd
from arize import ArizeClient

from workshop_helpers.backend import TOOLS, hydrate_backend_from_dataset, run_billing_agent_threadsafe
from workshop_helpers.metrics import pack_response_payload, build_evaluators
from workshop_helpers.scenarios import run_context_agent, run_raw_llm, run_router_structured

DATASET_NAME = "workiva-ai-support-copilot-benchmark"

VARIANT_BEHAVIORS = {
    "router": "Routing classifier. It should map the user question to exactly one supported support category.",
    "permissions": "Prompt-only permissions specialist. It answers from policy-style guidance without tools or hidden context.",
    "review_workflow": "Context-aware workflow specialist. It answers using retrieved workflow state and blocker context.",
    "billing": "Tool-using billing specialist. It uses tools and billing JSON guidance before acting or answering.",
    "v2_routed": "Two-stage support copilot. It routes first, then dispatches to the specialist matched to the routed category.",
}


def dataset_index(dataset: list[dict]) -> dict:
    return {case["scenario_id"]: case for case in dataset}


def select_cases(dataset: list[dict], limit_n: int | None = None, seed: int = 42) -> list[dict]:
    if limit_n is None:
        return list(dataset)
    shuffled = list(dataset)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit_n]


def select_cases_by_category(dataset: list[dict], category: str) -> list[dict]:
    return [case for case in dataset if case["category"] == category]


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


def build_review_context_message(customer_message: str, source_data: dict) -> str:
    context_block = json.dumps(source_data, indent=2)
    return f"Workflow context:\n{context_block}\n\nUser question:\n{customer_message}"


def dispatch_specialist_response(
    client,
    route_category: str,
    case: dict,
    prompt_permissions: str,
    prompt_review_workflow: str,
    prompt_billing: str,
    escalation_response_template: str,
) -> dict:
    user_input = case["user_input"]
    source_data = case["source_data"]
    router_record = {"name": "route_to_specialist", "arguments": json.dumps({"category": route_category})}

    if route_category == "permissions":
        response_text = run_raw_llm(client, user_input, prompt_permissions)
        return pack_response_payload(
            response_text,
            tool_calls=[router_record],
            metadata={"route_category": route_category},
        )

    if route_category == "review_workflow":
        response_text = run_context_agent(
            client,
            build_review_context_message(user_input, source_data),
            prompt_review_workflow,
        )
        return pack_response_payload(
            response_text,
            tool_calls=[router_record],
            metadata={"route_category": route_category},
        )

    if route_category == "billing":
        account_id = source_data.get("customer_id", "UNKNOWN")
        result = run_billing_agent_threadsafe(
            customer_message=user_input,
            instructions=prompt_billing.format(authenticated_account_id=account_id),
        )
        return pack_response_payload(
            result["output"],
            tool_calls=[router_record, *result.get("tool_calls", [])],
            action_calls=result.get("action_calls", []),
            metadata={"route_category": route_category},
        )

    response_text = escalation_response_template.format(
        account_name=source_data.get("account_name", "your team")
    )
    action_call = {
        "name": "escalate_to_human",
        "arguments": json.dumps({"customer_id": source_data.get("customer_id", "UNKNOWN"), "reason": case["user_input"]}),
    }
    return pack_response_payload(
        response_text,
        tool_calls=[router_record],
        action_calls=[action_call],
        metadata={"route_category": "escalation"},
    )


def build_tasks(
    client,
    dataset: list[dict],
    prompt_router: str,
    prompt_permissions: str,
    prompt_review_workflow: str,
    prompt_billing: str,
    escalation_response_template: str,
) -> dict:
    cases_by_id = dataset_index(dataset)
    categories = sorted({case["category"] for case in dataset})

    def task_router(row: dict) -> str:
        route = run_router_structured(client, row["user_input"], prompt_router, categories)
        return route["category"]

    def task_v2_routed(row: dict) -> str:
        case = cases_by_id.get(row["scenario_id"])
        if not case:
            return pack_response_payload("Error: case not found")
        route = run_router_structured(client, row["user_input"], prompt_router, categories)
        return dispatch_specialist_response(
            client=client,
            route_category=route["category"],
            case=case,
            prompt_permissions=prompt_permissions,
            prompt_review_workflow=prompt_review_workflow,
            prompt_billing=prompt_billing,
            escalation_response_template=escalation_response_template,
        )

    return {"task_router": task_router, "task_v2_routed": task_v2_routed}


def run_router_local(client, dataset: list[dict], prompt_router: str) -> pd.DataFrame:
    categories = sorted({case["category"] for case in dataset})
    rows = []
    for case in dataset:
        route = run_router_structured(client, case["user_input"], prompt_router, categories)
        predicted = route["category"]
        rows.append(
            {
                "scenario_id": case["scenario_id"],
                "expected_category": case["category"],
                "predicted_category": predicted,
                "exact_match": predicted == case["category"],
            }
        )
    return pd.DataFrame(rows)


def _find_score_column(results_df: pd.DataFrame) -> str | None:
    preferred = [col for col in results_df.columns if "exact" in col.lower() and "score" in col.lower()]
    if preferred:
        return preferred[0]
    generic = [col for col in results_df.columns if "score" in col.lower()]
    return generic[0] if generic else None


def summarize_router_experiment_results(results_df: pd.DataFrame) -> pd.DataFrame:
    score_col = _find_score_column(results_df)
    return pd.DataFrame(
        [
            {
                "rows_evaluated": len(results_df),
                "score_column": score_col or "not found",
                "exact_match_accuracy": results_df[score_col].mean() if score_col else None,
            }
        ]
    )


def summarize_experiment_scores(results_df: pd.DataFrame) -> pd.DataFrame:
    score_columns = [col for col in results_df.columns if "score" in col.lower()]
    if not score_columns:
        return pd.DataFrame(
            [
                {
                    "rows_evaluated": len(results_df),
                    "status": "No score columns found in the experiment results DataFrame.",
                }
            ]
        )

    summary = {"rows_evaluated": len(results_df)}
    for column in score_columns:
        numeric = pd.to_numeric(results_df[column], errors="coerce")
        if numeric.notna().any():
            summary[column] = numeric.mean()
    return pd.DataFrame([summary])


def run_experiment(arize_client, dataset_id: str, name_prefix: str, task, evaluators, concurrency: int = 1):
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
        ("router exact-match accuracy is stable", "check route experiment", "routing quality should be good before adding specialist autonomy"),
        ("brand voice remains strong", "check routed copilot experiment", "brand voice is a guardrail, not the primary success metric"),
        ("helpfulness stays high by category", "check routed copilot experiment", "specialists should answer appropriately for their support category"),
        ("human escalation is triggered when required", "check code evals", "high-risk or angry cases must hand off safely"),
        ("billing actions correspond to real workflows", "stubbed in demo backend", "verify this before any real pilot"),
        ("sample size is appropriate for workshop runtime", "configurable via LIMIT_N_CASES", "smaller runs are faster but noisier"),
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
    prompt_permissions: str,
    prompt_review_workflow: str,
    prompt_billing: str,
    escalation_response_template: str,
    judge_prompts: dict,
    limit_n: int | None = None,
    arize_client=None,
    dataset_id: str | None = None,
) -> dict:
    selected_dataset = select_cases(dataset, limit_n=limit_n)
    hydrate_backend_from_dataset(dataset)
    dataset_lookup = dataset_index(selected_dataset)
    if arize_client is not None and dataset_id is not None:
        arize_bundle = {
            "client": arize_client,
            "dataset_id": dataset_id,
            "dataset_name": DATASET_NAME,
            "row_count": len(selected_dataset),
            "created": False,
            "dataframe": build_arize_dataframe(selected_dataset),
        }
    else:
        arize_bundle = ensure_arize_dataset(arize_api_key, arize_space_id, selected_dataset)
    return {
        **arize_bundle,
        "dataset_lookup": dataset_lookup,
        "selected_dataset": selected_dataset,
        "tasks": build_tasks(
            client,
            selected_dataset,
            prompt_router,
            prompt_permissions,
            prompt_review_workflow,
            prompt_billing,
            escalation_response_template,
        ),
        "build_evaluators": lambda variant_name: build_evaluators(
            client,
            dataset_lookup,
            variant_name=variant_name,
            variant_behavior=VARIANT_BEHAVIORS[variant_name],
            judge_prompts=judge_prompts,
        ),
        "tool_count": len(TOOLS),
        "limit_n": limit_n,
    }
