from datetime import datetime
import json

import pandas as pd
from arize import ArizeClient

from workshop_helpers.backend import TOOLS, hydrate_backend_from_dataset, run_support_agent_threadsafe
from workshop_helpers.metrics import build_evaluators

DATASET_NAME = "cs-support-workshop-benchmark"

WORKSHOP_BENCHMARK_CONFIG = {
    "CS_E02": {
        "benchmark_slice": "prompt",
        "expected_behavior": "ask_one_targeted_followup",
        "v2_snapshot": {
            "customer_name": "Mark Davies",
            "customer_id": "CUST_3392",
            "account_status": "active",
        },
    },
    "CS_008": {
        "benchmark_slice": "prompt",
        "expected_behavior": "ask_one_targeted_followup",
        "v2_snapshot": {
            "customer_name": "Ben Hartley",
            "customer_id": "CUST_9920",
            "order_id": "ORD_5517",
            "order_status": "processing",
        },
    },
    "CS_004": {
        "benchmark_slice": "context",
        "expected_behavior": "answer_with_specific_context_no_action_claim",
        "v2_snapshot": {
            "customer_name": "James Okafor",
            "customer_id": "CUST_2290",
            "account_status": "active",
            "subscription_plan": "Premium Monthly",
            "subscription_price": 12.99,
            "last_billed": "2024-03-01",
            "subscription_origin": "trial auto-converted",
        },
    },
    "CS_007": {
        "benchmark_slice": "context",
        "expected_behavior": "answer_with_specific_context_no_action_claim",
        "v2_snapshot": {
            "customer_name": "Aisha Nguyen",
            "customer_id": "CUST_1145",
            "order_id": "ORD_6630",
            "product_name": "Yoga Mat Bundle",
            "order_status": "in_transit",
            "tracking": "TRK_882991",
            "estimated_delivery": "2024-03-24",
            "delay_reason": "weather hold at regional hub",
        },
    },
    "CS_009": {
        "benchmark_slice": "context",
        "expected_behavior": "answer_with_specific_context_no_action_claim",
        "v2_snapshot": {
            "customer_name": "Clara Johansson",
            "customer_id": "CUST_3388",
            "device": "Android 13",
            "app_version": "4.1.0",
            "latest_app_version": "4.2.1",
            "known_fix": "4.2.1 fixes order history crash on Android 13",
        },
    },
    "CS_011": {
        "benchmark_slice": "context",
        "expected_behavior": "answer_with_specific_context_no_action_claim",
        "v2_snapshot": {
            "customer_name": "Nina Kowalski",
            "customer_id": "CUST_4409",
            "product_name": "QuickCharge 15W Wireless Pad",
            "compatible_with": ["iPhone 8 and later", "all Android Qi devices"],
            "magsafe_compatible": True,
            "max_wattage_magsafe": 15,
            "max_wattage_qi": 7.5,
        },
    },
    "CS_003": {
        "benchmark_slice": "tools",
        "expected_behavior": "confirm_backend_action_taken",
        "v2_snapshot": {
            "customer_name": "Sophie Williams",
            "customer_id": "CUST_6614",
            "order_id": "ORD_7753",
            "product_name": "Ceramic Pour-Over Coffee Set",
            "order_status": "processing",
            "reported_issue": "customer reports duplicate charge",
        },
    },
    "CS_015": {
        "benchmark_slice": "tools",
        "expected_behavior": "confirm_backend_action_taken",
        "v2_snapshot": {
            "customer_name": "Jack Morrison",
            "customer_id": "CUST_5540",
            "order_id": "ORD_3318",
            "product_name": "Classic Canvas Tote - Navy Blue",
            "order_status": "delivered",
            "reported_issue": "customer says the wrong colour arrived",
        },
    },
    "CS_E04": {
        "benchmark_slice": "tools",
        "expected_behavior": "confirm_backend_action_taken",
        "v2_snapshot": {
            "customer_name": "Carlos Mendez",
            "customer_id": "CUST_9934",
            "order_id": "ORD_7701",
            "product_name": "Smart Home Starter Kit",
            "order_status": "lost_in_transit",
            "prior_contacts": 3,
            "reported_issue": "customer is demanding a refund for a missing package",
        },
    },
}


def dataset_index(dataset: list[dict]) -> dict:
    return {case["scenario_id"]: case for case in dataset}


def build_workshop_benchmark(dataset: list[dict]) -> list[dict]:
    indexed = dataset_index(dataset)
    benchmark = []
    for scenario_id, metadata in WORKSHOP_BENCHMARK_CONFIG.items():
        case = dict(indexed[scenario_id])
        case["benchmark_slice"] = metadata["benchmark_slice"]
        case["expected_behavior"] = metadata["expected_behavior"]
        case["v2_snapshot"] = metadata["v2_snapshot"]
        benchmark.append(case)
    return benchmark


def summarize_dataset(dataset: list[dict]) -> dict:
    frame = pd.DataFrame(dataset)
    summary = {
        "scenario_count": len(frame),
        "categories": sorted(frame.category.unique()),
    }
    if "is_edge_case" in frame.columns:
        summary["standard_count"] = int(frame[~frame.is_edge_case].shape[0])
        summary["edge_case_count"] = int(frame[frame.is_edge_case].shape[0])
    if "benchmark_slice" in frame.columns:
        summary["slice_counts"] = frame["benchmark_slice"].value_counts().sort_index().to_dict()
    return summary


def build_v2_support_snapshot(case: dict) -> dict:
    return dict(case.get("v2_snapshot") or WORKSHOP_BENCHMARK_CONFIG.get(case["scenario_id"], {}).get("v2_snapshot", {}))


def build_arize_dataframe(dataset: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": case["scenario_id"],
                "category": case["category"],
                "benchmark_slice": case["benchmark_slice"],
                "expected_behavior": case["expected_behavior"],
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


def build_tasks(client, dataset: list[dict], prompt_v1: str, prompt_v2: str, prompt_v3: str) -> dict:
    cases_by_id = dataset_index(dataset)

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
        return response.choices[0].message.content.strip()

    def task_v2(row: dict) -> str:
        case = cases_by_id.get(row["scenario_id"])
        if not case:
            return "Error: case not found"
        support_snapshot = build_v2_support_snapshot(case)
        message = (
            f"Customer context (internal support snapshot):\n{json.dumps(support_snapshot, indent=2)}"
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
        return response.choices[0].message.content.strip()

    def task_v3(row: dict) -> str:
        return run_support_agent_threadsafe(
            customer_message=row["user_input"],
            authenticated_customer_id=row.get("customer_id") or "UNKNOWN",
            instructions=prompt_v3.format(
                authenticated_customer_id=row.get("customer_id") or "UNKNOWN"
            ),
        )

    return {"task_v1": task_v1, "task_v2": task_v2, "task_v3": task_v3}


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
        ("correct_outcome >= 80% on tools slice", "check v3 benchmark in Arize", "tool-using agent should outperform static context on operational cases"),
        ("workflow_fit >= 80% on all slices", "check slice filters in Arize", "prompt, context, and tools should each behave as designed"),
        ("tone Good or Acceptable >= 85%", "check all three variants", "tone is a guardrail, not the main success metric"),
        ("human review gate before response sent to customer", "not yet designed", "required for v1: agent drafts, human approves"),
        ("guardrail for escalation-required cases", "take_action(escalate) returns a ticket", "verify downstream escalation workflow is real"),
        ("tool coverage matches benchmark cases", "yes for workshop subset", "expand only after the story is clear"),
        ("sampling plan: outputs reviewed per week, by whom", "not defined", "recommended: 30 benchmark rechecks per prompt change"),
        ("threshold to advance from demo to pilot", "not defined", "suggested: strong tools-slice performance plus stable tone guardrail"),
    ]


def format_checklist_rows(checklist: list[tuple[str, str, str]]) -> list[str]:
    rows = [f"{'#':<3} {'CRITERION':<44} {'STATUS':<34} NOTE", "-" * 118]
    for index, (criterion, status, note) in enumerate(checklist, start=1):
        icon = "OK" if any(word in status.lower() for word in ["yes", "check", "returns"]) else "--"
        rows.append(f"{icon} {index:<2} {criterion:<44} {status:<34} {note}")
    return rows


def prepare_experiment_bundle(
    client,
    arize_api_key: str,
    arize_space_id: str,
    dataset: list[dict],
    prompt_v1: str,
    prompt_v2: str,
    prompt_v3: str,
) -> dict:
    hydrate_backend_from_dataset(dataset)
    dataset_lookup = dataset_index(dataset)
    arize_bundle = ensure_arize_dataset(arize_api_key, arize_space_id, dataset)
    return {
        **arize_bundle,
        "dataset_lookup": dataset_lookup,
        "tasks": build_tasks(client, dataset, prompt_v1, prompt_v2, prompt_v3),
        "evaluators": build_evaluators(client, dataset_lookup),
        "tool_count": len(TOOLS),
    }
