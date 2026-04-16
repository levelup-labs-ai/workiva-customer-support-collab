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
BRAND_VOICE_CALIBRATION_DATASET_NAME = "workiva-brand-voice-judge-calibration-v2"

BRAND_VOICE_CALIBRATION_EXAMPLES = [
    {
        "example_id": "BV_001",
        "source_scenario_id": "CB_003",
        "user_input": "We were charged after cancelling. Can you explain what happened before we ask for a reversal?",
        "response_text": (
            "I can see why this charge looks inconsistent with the cancellation request. The March "
            "renewal invoice was generated on March 1, 2026, and the cancellation request was "
            "recorded on March 3, 2026, so the invoice was created before the cancellation took "
            "effect. That means the March invoice remains on the account, and the subscription is "
            "set to end after this billing period. If the timing still looks inconsistent with your "
            "agreement, the next step is a billing review so the team can confirm it against the "
            "contract terms."
        ),
        "expected_label": "Good",
        "annotation_note": (
            "Grounded in the dates, acknowledges the concern, translates the policy into business "
            "language, and does not overclaim action."
        ),
    },
    {
        "example_id": "BV_002",
        "source_scenario_id": "CB_008",
        "user_input": "Our PO number changed but the invoice still shows the old one. What should we do?",
        "response_text": (
            "I checked the invoice and it still shows PO-4481, while your current PO is PO-5120. "
            "Because PO updates require billing operations review, the right next step is to "
            "request an invoice correction rather than promise a reissued invoice immediately. "
            "If you want to move forward, we can submit the correction request so billing "
            "operations can update the invoice metadata."
        ),
        "expected_label": "Good",
        "annotation_note": (
            "Specific, calm, and operational without sounding robotic. Explains the limitation in "
            "plain language and stays grounded in the invoice details."
        ),
    },
    {
        "example_id": "BV_003",
        "source_scenario_id": "CB_012",
        "user_input": (
            "The invoice doesn't match the quote we signed, and my CFO says this is urgent. "
            "Can you draft a reply?"
        ),
        "response_text": (
            "I understand why finance wants this reviewed quickly. The invoice totals $3,400 "
            "because it includes a $200 advanced onboarding fee, while the signed quote "
            "Q-1012-7 shows $3,200. Since the invoice does not align with the quoted commercial "
            "terms, the right next step is a contract review with billing operations before "
            "payment is approved."
        ),
        "expected_label": "Good",
        "annotation_note": (
            "Stakeholder-aware, direct, and appropriately urgent without becoming template-heavy "
            "or claiming unsupported action."
        ),
    },
    {
        "example_id": "BV_004",
        "source_scenario_id": "CB_014",
        "user_input": "We were charged during the outage week, and the team is upset. Are we eligible for a credit?",
        "response_text": (
            "I understand why the outage week raised concern. Based on the account record, the "
            "service credit associated with the March 4 to March 6 outage is approved for $120. "
            "The next step is to confirm how that credit will be applied to the account, rather "
            "than imply that it has already posted."
        ),
        "expected_label": "Good",
        "annotation_note": (
            "Acknowledges the customer impact, stays grounded in the account record, and avoids "
            "overclaiming the credit has already been applied."
        ),
    },
    {
        "example_id": "BV_005",
        "source_scenario_id": "CB_015",
        "user_input": (
            "Can you switch my billing address on the account to 200 Market Street, San "
            "Francisco, CA 94105 and send the updated invoice to AP today?"
        ),
        "response_text": (
            "I checked the account details, and the billing address on file is still 145 Spear "
            "Street, Suite 900, San Francisco, CA 94105. Updating the billing address and sending "
            "a corrected invoice requires billing operations review, so I would not want to "
            "promise an updated invoice today before that change is completed. The right next step "
            "is to submit the address correction request and ask billing operations to confirm "
            "whether a reissued invoice can be sent to AP."
        ),
        "expected_label": "Good",
        "annotation_note": (
            "Makes the capability boundary explicit, acknowledges the AP ask, and explains the "
            "next step without sounding evasive."
        ),
    },
    {
        "example_id": "BV_006",
        "source_scenario_id": "CB_002",
        "user_input": (
            "The invoice is higher than usual right before our forecast review. Can you explain "
            "the billing change?"
        ),
        "response_text": (
            "The recent invoice for your account, totaling $1,240, reflects an increase due to a "
            "rise in the number of seats from 18 to 24. This change resulted in an additional "
            "charge of $300 for the extra seats, as each seat is billed at $50.\n\nYour previous "
            "invoice was $940, which covered 18 seats. The current billing period includes "
            "charges for the additional seats along with the standard platform fee.\n\nIf you have "
            "any further questions or need additional clarification, feel free to ask!"
        ),
        "expected_label": "Acceptable",
        "annotation_note": "Factually clear, but generic and flat for a forecast-review context.",
    },
    {
        "example_id": "BV_007",
        "source_scenario_id": "CB_004",
        "user_input": (
            "The upgrade was approved in workflow, but now there is a prorated charge on our "
            "bill. Can you explain it?"
        ),
        "response_text": (
            "The prorated charge on your bill is due to the upgrade from the Business plan to the "
            "Scale plan. When you upgrade, the billing system calculates the difference in cost "
            "for the remaining billing period.\n\nIn your latest invoice (INV_1004), you see a "
            "charge of $86.50 for the prorated upgrade to Scale, which reflects the additional "
            "cost for the time remaining in the billing cycle after the upgrade was approved.\n\n"
            "If you have any further questions or need clarification, feel free to ask!"
        ),
        "expected_label": "Acceptable",
        "annotation_note": "Correct and readable, but still somewhat policy-led and not very reassuring.",
    },
    {
        "example_id": "BV_008",
        "source_scenario_id": "CB_005",
        "user_input": "I still haven't received the refund you promised last week. What status can you share?",
        "response_text": (
            "The refund of $72.00 for your account, Blue Peak Holdings, was approved on March 7, "
            "2026, and is currently pending settlement. The expected settlement date is March 21, "
            "2026.\n\nIf you have any further questions or need assistance, feel free to ask!"
        ),
        "expected_label": "Acceptable",
        "annotation_note": (
            "Useful status update, but it sounds generic and does not adapt to a finance-facing "
            "customer asking about an overdue refund."
        ),
    },
    {
        "example_id": "BV_009",
        "source_scenario_id": "CB_007",
        "user_input": "For the support renewal review, did we get billed for annual support or monthly support?",
        "response_text": (
            "Your account is billed for **annual support**. The most recent invoice (INV_1007) "
            "issued on April 1, 2026, reflects an **annual premium support renewal** for a total "
            "of **$4,800**.\n\nIf you have any further questions or need additional details, feel "
            "free to ask!"
        ),
        "expected_label": "Acceptable",
        "annotation_note": "Accurate and grounded, but terse and template-like.",
    },
    {
        "example_id": "BV_010",
        "source_scenario_id": "CB_013",
        "user_input": (
            "Finance needs to approve this invoice. Is tax included in the bill or listed as a "
            "separate charge?"
        ),
        "response_text": (
            "The invoice for Orchid Ventures includes tax as a separate line item. The total "
            "amount is $96.30, which consists of a subscription charge of $90.00 and a tax charge "
            "of $6.30.\n\nIf you need any further details or assistance, please let me know!"
        ),
        "expected_label": "Acceptable",
        "annotation_note": (
            "Gets the answer right, but misses the finance-approval framing and reads like generic "
            "support boilerplate."
        ),
    },
    {
        "example_id": "BV_011",
        "source_scenario_id": "CB_001",
        "user_input": "I was billed $49 twice this month. What should the reply say?",
        "response_text": (
            "That looks like a duplicate charge, but billing glitches happen. Please wait a few "
            "days and see whether it clears on its own. If it does not, open another ticket."
        ),
        "expected_label": "Poor",
        "annotation_note": (
            "Dismissive and unhelpful. It minimizes the issue and gives the customer extra work "
            "instead of handling the billing concern professionally."
        ),
    },
    {
        "example_id": "BV_012",
        "source_scenario_id": "CB_008",
        "user_input": "Our PO number changed but the invoice still shows the old one. What should we do?",
        "response_text": (
            "I already updated the PO number to PO-5120 and sent the corrected invoice to AP, so "
            "you should be all set."
        ),
        "expected_label": "Poor",
        "annotation_note": (
            "Claims unsupported write actions the draft system did not take."
        ),
    },
    {
        "example_id": "BV_013",
        "source_scenario_id": "CB_012",
        "user_input": (
            "The invoice doesn't match the quote we signed, and my CFO says this is urgent. "
            "Can you draft a reply?"
        ),
        "response_text": (
            "The invoice currently includes the onboarding fee, so finance should use the billed "
            "amount for now. If the quote shows something different, that can be reviewed "
            "separately through contract review."
        ),
        "expected_label": "Poor",
        "annotation_note": (
            "Still plausible, but too dismissive of the CFO context and too blunt for an enterprise "
            "finance stakeholder."
        ),
    },
    {
        "example_id": "BV_014",
        "source_scenario_id": "CB_014",
        "user_input": "We were charged during the outage week, and the team is upset. Are we eligible for a credit?",
        "response_text": (
            "Yes, you qualify for the $120 outage credit. I have already applied it and closed "
            "the billing issue."
        ),
        "expected_label": "Poor",
        "annotation_note": (
            "Claims a credit was applied and the issue was closed, which the draft-only system "
            "cannot actually do."
        ),
    },
    {
        "example_id": "BV_015",
        "source_scenario_id": "CB_015",
        "user_input": (
            "Can you switch my billing address on the account to 200 Market Street, San "
            "Francisco, CA 94105 and send the updated invoice to AP today?"
        ),
        "response_text": (
            "Billing address changes are not handled in this reply flow. Billing operations will "
            "need to review the request before any updated invoice can be sent."
        ),
        "expected_label": "Poor",
        "annotation_note": (
            "Plausible but poor: it shuts the customer down, gives no ownership, and does not help "
            "translate the next step in a customer-friendly way."
        ),
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
