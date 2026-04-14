from datetime import datetime
import json
import random

import pandas as pd
from arize import ArizeClient

from workshop_helpers.backend import TOOLS, hydrate_backend_from_dataset, run_billing_agent_threadsafe
from workshop_helpers.metrics import pack_response_payload, build_evaluators
from workshop_helpers.scenarios import run_context_agent, run_raw_llm, run_router_raw

DATASET_NAME = "workiva-ai-support-copilot-benchmark"
V2_BILLING_DATASET_NAME = "workiva-v2-billing-draft-benchmark"
V2_ALL_SPECIALISTS_DATASET_NAME = "workiva-v2-all-specialists-draft-benchmark"
BRAND_VOICE_CALIBRATION_DATASET_NAME = "workiva-brand-voice-judge-calibration"

BRAND_VOICE_CALIBRATION_EXAMPLES = [
    {
        "example_id": "BV_001",
        "response_text": "I checked the invoice and found the account appears eligible for a billing credit. If you confirm, support can apply the credit to the account.",
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
        "response_text": "I understand why the duplicate charge is frustrating. The account data suggests this may qualify for a credit, and I can ask the billing team to apply it after you confirm.",
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

VARIANT_BEHAVIORS = {
    "router": "Routing classifier. It should map the user question to exactly one supported support category.",
    "permissions": "Prompt-only permissions specialist. It answers from policy-style guidance without tools or hidden context.",
    "review_workflow": "Context-aware workflow specialist. It answers using retrieved workflow state and blocker context.",
    "billing": "Tool-using billing specialist. It uses read-only tools and billing JSON guidance before drafting an answer.",
    "v2_routed": "Two-stage support copilot. It routes first, then dispatches to the specialist matched to the routed category to draft a reply.",
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


def select_cases_by_categories(dataset: list[dict], categories: list[str]) -> list[dict]:
    category_set = set(categories)
    return [case for case in dataset if case["category"] in category_set]


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


def ensure_arize_dataset(
    arize_api_key: str,
    arize_space_id: str,
    dataset: list[dict],
    dataset_name: str = DATASET_NAME,
) -> dict:
    client = ArizeClient(api_key=arize_api_key)
    dataset_frame = build_arize_dataframe(dataset)
    list_response = client.datasets.list(space=arize_space_id)

    existing = next((item for item in list_response.datasets if item.name == dataset_name), None)
    if existing:
        dataset_id = existing.id
        created = False
    else:
        response = client.datasets.create(
            space=arize_space_id,
            name=dataset_name,
            examples=dataset_frame,
        )
        dataset_id = response.id
        created = True

    return {
        "client": client,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "row_count": len(dataset_frame),
        "created": created,
        "dataframe": dataset_frame,
    }


def build_brand_voice_calibration_dataframe() -> pd.DataFrame:
    return pd.DataFrame(BRAND_VOICE_CALIBRATION_EXAMPLES)


def ensure_brand_voice_calibration_dataset(arize_api_key: str, arize_space_id: str) -> dict:
    client = ArizeClient(api_key=arize_api_key)
    dataset_frame = build_brand_voice_calibration_dataframe()
    list_response = client.datasets.list(space=arize_space_id)

    existing = next(
        (item for item in list_response.datasets if item.name == BRAND_VOICE_CALIBRATION_DATASET_NAME),
        None,
    )
    if existing:
        dataset_id = existing.id
        created = False
    else:
        response = client.datasets.create(
            space=arize_space_id,
            name=BRAND_VOICE_CALIBRATION_DATASET_NAME,
            examples=dataset_frame,
        )
        dataset_id = response.id
        created = True

    return {
        "client": client,
        "dataset_id": dataset_id,
        "dataset_name": BRAND_VOICE_CALIBRATION_DATASET_NAME,
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
    return pack_response_payload(
        response_text,
        tool_calls=[router_record],
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
    routing_categories: list[str] | None = None,
) -> dict:
    cases_by_id = dataset_index(dataset)
    categories = routing_categories or sorted({case["category"] for case in dataset})

    def task_router(row: dict) -> str:
        route = run_router_raw(client, row["user_input"], prompt_router, categories)
        return route["category"]

    def task_v2_routed(row: dict) -> str:
        case = cases_by_id.get(row["scenario_id"])
        if not case:
            return pack_response_payload("Error: case not found")
        route = run_router_raw(client, row["user_input"], prompt_router, categories)
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
        route = run_router_raw(client, case["user_input"], prompt_router, categories)
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
        ("human escalation language is clear", "inspect routed traces", "high-risk or angry cases should draft a human handoff"),
        ("billing tools answer the right question", "inspect billing traces", "add or fix read-only tools when the draft lacks account facts"),
        ("brand voice judge is calibrated", "check calibration dataset", "judge labels should match human annotations before reuse"),
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
    dataset_name: str = DATASET_NAME,
    routing_categories: list[str] | None = None,
) -> dict:
    selected_dataset = select_cases(dataset, limit_n=limit_n)
    hydrate_backend_from_dataset(dataset)
    dataset_lookup = dataset_index(selected_dataset)
    if arize_client is not None and dataset_id is not None:
        arize_bundle = {
            "client": arize_client,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "row_count": len(selected_dataset),
            "created": False,
            "dataframe": build_arize_dataframe(selected_dataset),
        }
    else:
        arize_bundle = ensure_arize_dataset(
            arize_api_key,
            arize_space_id,
            selected_dataset,
            dataset_name=dataset_name,
        )
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
            routing_categories=routing_categories,
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
