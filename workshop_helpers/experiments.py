from datetime import datetime
import json
import random

import pandas as pd
from arize import ArizeClient

from workshop_helpers.backend import hydrate_backend_from_dataset, run_billing_agent_threadsafe
from workshop_helpers.metrics import pack_response_payload, build_evaluators
from workshop_helpers.scenarios import run_context_agent, run_raw_llm, run_router_raw

DATASET_NAME = "workiva-ai-support-copilot-benchmark"
V2_BILLING_DATASET_NAME = "workiva-v2-billing-draft-benchmark"
V2_ALL_SPECIALISTS_DATASET_NAME = "workiva-v2-all-specialists-draft-benchmark"
BRAND_VOICE_CALIBRATION_DATASET_NAME = "workiva-brand-voice-judge-calibration"

BRAND_VOICE_CALIBRATION_EXAMPLES = [
    {
        "example_id": "BV_001",
        "response_text": (
            "I checked the invoice and found the account appears eligible for a billing credit. "
            "If you confirm, support can apply the credit to the account."
        ),
        "expected_label": "Good",
        "annotation_note": "Clear, calm, and does not claim to take the write action.",
    },
    {
        "example_id": "BV_002",
        "response_text": "This is probably just how billing works. Contact someone else if you need it fixed.",
        "expected_label": "Poor",
        "annotation_note": "Dismissive and not useful.",
    },
    {
        "example_id": "BV_003",
        "response_text": (
            "I understand why the duplicate charge is frustrating. The account data suggests this "
            "may qualify for a credit, and I can ask the billing team to apply it after you confirm."
        ),
        "expected_label": "Good",
        "annotation_note": "Empathetic and careful about action permission.",
    },
    {
        "example_id": "BV_004",
        "response_text": "Your invoice was higher because of added seats. Let us know if you need anything else.",
        "expected_label": "Acceptable",
        "annotation_note": "Accurate but a little terse and generic.",
    },
    {
        "example_id": "BV_005",
        "response_text": "I already fixed the invoice and escalated the case.",
        "expected_label": "Poor",
        "annotation_note": "Claims write actions that the draft-reply system should not take.",
    },
]

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


def select_cases_by_categories(dataset: list[dict], categories: list[str]) -> list[dict]:
    category_set = set(categories)
    return [case for case in dataset if case["category"] in category_set]


def summarize_dataset(dataset: list[dict]) -> dict:
    frame = pd.DataFrame(dataset)
    return {
        "scenario_count": len(frame),
        "categories": sorted(frame.category.unique()),
    }


def build_arize_dataframe(dataset: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": case["scenario_id"],
                "category": case["category"],
                "customer_id": case["source_data"].get("customer_id", ""),
                "user_input": case["user_input"],
            }
            for case in dataset
        ]
    )


def _dataset_bundle(client, dataset_id: str, dataset_name: str, dataset_frame: pd.DataFrame, created: bool) -> dict:
    return {
        "client": client,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "row_count": len(dataset_frame),
        "created": created,
        "dataframe": dataset_frame,
    }


def _ensure_dataframe_dataset(client, arize_space_id: str, dataset_name: str, dataset_frame: pd.DataFrame) -> dict:
    list_response = client.datasets.list(space=arize_space_id)
    existing = next((item for item in list_response.datasets if item.name == dataset_name), None)
    if existing:
        return _dataset_bundle(client, existing.id, dataset_name, dataset_frame, created=False)

    response = client.datasets.create(
        space=arize_space_id,
        name=dataset_name,
        examples=dataset_frame,
    )
    return _dataset_bundle(client, response.id, dataset_name, dataset_frame, created=True)


def ensure_arize_dataset(
    arize_api_key: str,
    arize_space_id: str,
    dataset: list[dict],
    dataset_name: str = DATASET_NAME,
) -> dict:
    client = ArizeClient(api_key=arize_api_key)
    dataset_frame = build_arize_dataframe(dataset)
    return _ensure_dataframe_dataset(client, arize_space_id, dataset_name, dataset_frame)


def build_brand_voice_calibration_dataframe() -> pd.DataFrame:
    return pd.DataFrame(BRAND_VOICE_CALIBRATION_EXAMPLES)


def ensure_brand_voice_calibration_dataset(arize_api_key: str, arize_space_id: str) -> dict:
    client = ArizeClient(api_key=arize_api_key)
    dataset_frame = build_brand_voice_calibration_dataframe()
    return _ensure_dataframe_dataset(
        client,
        arize_space_id,
        BRAND_VOICE_CALIBRATION_DATASET_NAME,
        dataset_frame,
    )


def build_review_context_message(customer_message: str, source_data: dict) -> str:
    context_block = json.dumps(source_data, indent=2)
    return f"Workflow context:\n{context_block}\n\nUser question:\n{customer_message}"


def _route_record(route_category: str) -> dict:
    return {"name": "route_to_specialist", "arguments": json.dumps({"category": route_category})}


def _pack_routed_response(
    route_category: str,
    response_text: str,
    *,
    tool_calls: list | None = None,
    action_calls: list | None = None,
) -> str:
    return pack_response_payload(
        response_text,
        tool_calls=tool_calls,
        action_calls=action_calls,
        metadata={"route_category": route_category},
    )


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
    router_record = _route_record(route_category)

    if route_category == "permissions":
        return _pack_routed_response(
            route_category,
            run_raw_llm(client, user_input, prompt_permissions),
            tool_calls=[router_record],
        )

    if route_category == "review_workflow":
        return _pack_routed_response(
            route_category,
            run_context_agent(
                client,
                build_review_context_message(user_input, source_data),
                prompt_review_workflow,
            ),
            tool_calls=[router_record],
        )

    if route_category == "billing":
        account_id = source_data.get("customer_id", "UNKNOWN")
        result = run_billing_agent_threadsafe(
            customer_message=user_input,
            instructions=prompt_billing.format(authenticated_account_id=account_id),
        )
        return _pack_routed_response(
            route_category,
            result["output"],
            tool_calls=[router_record, *result.get("tool_calls", [])],
            action_calls=result.get("action_calls", []),
        )

    return _pack_routed_response(
        "escalation",
        escalation_response_template.format(account_name=source_data.get("account_name", "your team")),
        tool_calls=[router_record],
    )


def _routing_categories(dataset: list[dict], explicit_categories: list[str] | None = None) -> list[str]:
    return explicit_categories or sorted({case["category"] for case in dataset})


def _route_category(client, user_input: str, prompt_router: str, categories: list[str]) -> str:
    return run_router_raw(client, user_input, prompt_router, categories)["category"]


def build_tasks(
    client,
    dataset: list[dict],
    prompt_router: str,
    prompt_permissions: str,
    prompt_review_workflow: str,
    prompt_billing: str,
    escalation_response_template: str,
    routing_categories: list[str] | None = None,
) -> dict:
    cases_by_id = dataset_index(dataset)
    categories = _routing_categories(dataset, routing_categories)

    def task_router(row: dict) -> str:
        return _route_category(client, row["user_input"], prompt_router, categories)

    def task_v2_routed(row: dict) -> str:
        case = cases_by_id.get(row["scenario_id"])
        if not case:
            return pack_response_payload("Error: case not found")
        return dispatch_specialist_response(
            client=client,
            route_category=_route_category(client, row["user_input"], prompt_router, categories),
            case=case,
            prompt_permissions=prompt_permissions,
            prompt_review_workflow=prompt_review_workflow,
            prompt_billing=prompt_billing,
            escalation_response_template=escalation_response_template,
        )

    return {"task_router": task_router, "task_v2_routed": task_v2_routed}


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
    dataset_name: str = DATASET_NAME,
    routing_categories: list[str] | None = None,
) -> dict:
    selected_dataset = select_cases(dataset, limit_n=limit_n)
    hydrate_backend_from_dataset(dataset)
    dataset_frame = build_arize_dataframe(selected_dataset)
    if arize_client is not None and dataset_id is not None:
        arize_bundle = _dataset_bundle(arize_client, dataset_id, dataset_name, dataset_frame, created=False)
    else:
        arize_bundle = ensure_arize_dataset(
            arize_api_key,
            arize_space_id,
            selected_dataset,
            dataset_name=dataset_name,
        )
    return {
        **arize_bundle,
        "tasks": build_tasks(
            client,
            selected_dataset,
            prompt_router,
            prompt_permissions,
            prompt_review_workflow,
            prompt_billing,
            escalation_response_template,
            routing_categories=routing_categories,
        ),
        "build_evaluators": lambda variant_name: build_evaluators(
            client,
            variant_name=variant_name,
            judge_prompts=judge_prompts,
        ),
    }
