import asyncio
import concurrent.futures
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast, get_args

from agents import Agent, ModelSettings, Runner, function_tool

from workshop_helpers.data import (
    BillingCase,
    BillingInvoiceSourceData,
    BillingLineItemSourceData,
    BillingSourceData,
    EscalationCase,
    EscalationSourceData,
    SupportCase,
)
from workshop_helpers.setup import suspend_openai_tracing_for_agents

_BILLING_REFERENCE_PATH = Path(__file__).with_name("billing_reference.json")

BillingPolicyIssueType = Literal[
    "charge_explanation",
    "credit_or_waiver_review",
    "refund_status",
    "invoice_update_request",
    "contract_or_cancellation_review",
]
BILLING_POLICY_ISSUE_TYPES = cast(tuple[str, ...], get_args(BillingPolicyIssueType))


@dataclass(frozen=True)
class BillingAccountRecord:
    account_name: str
    plan_name: str
    billing_status: str
    credit_eligible: bool
    account_context: dict[str, object]

    @classmethod
    def from_billing_source(cls, source_data: BillingSourceData) -> "BillingAccountRecord":
        return cls(
            account_name=source_data["account_name"],
            plan_name=source_data["plan_name"],
            billing_status=source_data["billing_status"],
            credit_eligible=source_data["credit_eligible"],
            account_context=source_data["account_context"],
        )

    @classmethod
    def from_escalation_source(cls, source_data: EscalationSourceData) -> "BillingAccountRecord":
        # Escalation cases seed the account lookup table even when there is no invoice metadata.
        return cls(
            account_name=source_data["account_name"],
            plan_name="Unknown",
            billing_status="active",
            credit_eligible=False,
            account_context={},
        )

    def to_tool_payload(self, account_id: str) -> dict:
        return {"account_id": account_id, **asdict(self)}


@dataclass(frozen=True)
class InvoiceLineItemRecord:
    description: str
    amount: float
    quantity: float
    unit_price: float

    @classmethod
    def from_source(cls, source_data: BillingLineItemSourceData) -> "InvoiceLineItemRecord":
        return cls(
            description=source_data["description"],
            amount=source_data["amount"],
            quantity=source_data["quantity"],
            unit_price=source_data["unit_price"],
        )


@dataclass(frozen=True)
class InvoiceRecord:
    invoice_id: str
    issued_on: str
    due_on: str
    billing_period_start: str
    billing_period_end: str
    status: str
    total_amount: float
    currency: str
    po_number: str
    tax_amount: float
    line_items: list[InvoiceLineItemRecord]

    @classmethod
    def from_source(cls, source_data: BillingInvoiceSourceData) -> "InvoiceRecord":
        return cls(
            invoice_id=source_data["invoice_id"],
            issued_on=source_data["issued_on"],
            due_on=source_data["due_on"],
            billing_period_start=source_data["billing_period_start"],
            billing_period_end=source_data["billing_period_end"],
            status=source_data["status"],
            total_amount=source_data["total_amount"],
            currency=source_data["currency"],
            po_number=source_data["po_number"],
            tax_amount=source_data["tax_amount"],
            line_items=[InvoiceLineItemRecord.from_source(item) for item in source_data["line_items"]],
        )

    def to_tool_payload(self) -> dict:
        return asdict(self)


class BackendSnapshot(TypedDict):
    billing_account_count: int
    invoice_count: int
    billing_reference_topics: int


BILLING_ACCOUNT_DB: dict[str, BillingAccountRecord] = {}
ACCOUNT_INVOICE_DB: dict[str, list[InvoiceRecord]] = {}
BILLING_REFERENCE_DB: dict[str, dict] = (
    json.loads(_BILLING_REFERENCE_PATH.read_text()) if _BILLING_REFERENCE_PATH.exists() else {}
)
_missing_policy_topics = set(BILLING_POLICY_ISSUE_TYPES) - set(BILLING_REFERENCE_DB)
_unexpected_policy_topics = set(BILLING_REFERENCE_DB) - set(BILLING_POLICY_ISSUE_TYPES)
if _missing_policy_topics or _unexpected_policy_topics:
    raise RuntimeError(
        "billing_reference.json is out of sync with BillingPolicyIssueType: "
        f"missing={sorted(_missing_policy_topics)} unexpected={sorted(_unexpected_policy_topics)}"
    )


def snapshot_backend() -> BackendSnapshot:
    return {
        "billing_account_count": len(BILLING_ACCOUNT_DB),
        "invoice_count": sum(len(invoices) for invoices in ACCOUNT_INVOICE_DB.values()),
        "billing_reference_topics": len(BILLING_REFERENCE_DB),
    }


@function_tool
def get_billing_account(account_id: str) -> dict:
    """Look up the billing account record for a customer or workspace after reading the billing policy for the issue."""
    account = BILLING_ACCOUNT_DB.get(account_id)
    if not account:
        return {"error": f"Billing account not found: {account_id}"}
    return account.to_tool_payload(account_id)


@function_tool
def list_invoices(account_id: str) -> dict:
    """List invoices for a billing account, including totals and structured line items.

    Read the billing policy for the issue before calling this tool.
    """
    invoices = ACCOUNT_INVOICE_DB.get(account_id)
    if invoices is None:
        return {"error": f"Billing account not found: {account_id}"}
    return {"account_id": account_id, "invoices": [invoice.to_tool_payload() for invoice in invoices]}


@function_tool
def get_billing_policy(issue_type: BillingPolicyIssueType) -> dict:
    """Call this first before any other billing tool to retrieve the applicable billing policy.

    Supported issue_type values are:
    - charge_explanation: invoice increases, proration, annual renewals, overages, or tax display questions
    - credit_or_waiver_review: duplicate charges, late-fee removals, setup-fee waivers, or outage/service credits
    - refund_status: the status of an already-approved refund
    - invoice_update_request: PO number or invoice metadata corrections
    - contract_or_cancellation_review: post-cancellation charges or mismatches with signed terms

    If the user request overlaps multiple categories, choose the closest primary issue.
    """
    return {
        "issue_type": issue_type,
        "policy": BILLING_REFERENCE_DB[issue_type],
        "supported_issue_types": list(BILLING_POLICY_ISSUE_TYPES),
    }


TOOLS = [
    get_billing_policy,
    get_billing_account,
    list_invoices,
]


def build_billing_agent(
    model: str = "gpt-4o-mini",
    instructions: str | None = None,
) -> Agent:
    return Agent(
        name="Billing Support Agent",
        instructions=instructions,
        tools=TOOLS,
        model=model,
        model_settings=ModelSettings(temperature=0, parallel_tool_calls=False),
    )


async def run_billing_agent_async(
    customer_message: str,
    model: str = "gpt-4o-mini",
    instructions: str | None = None,
):
    agent = build_billing_agent(model=model, instructions=instructions)
    return await Runner.run(agent, customer_message)


def run_billing_agent(
    customer_message: str,
    model: str = "gpt-4o-mini",
    instructions: str | None = None,
) -> dict:
    with suspend_openai_tracing_for_agents():
        result = asyncio.run(
            run_billing_agent_async(
                customer_message=customer_message,
                model=model,
                instructions=instructions,
            )
        )
    tool_calls = []
    for item in result.new_items:
        raw = getattr(item, "raw_item", None)
        if not raw or not hasattr(raw, "name"):
            continue
        entry = {"name": raw.name}
        arguments = getattr(raw, "arguments", None)
        if arguments is not None:
            entry["arguments"] = arguments
        tool_calls.append(entry)

    return {
        "output": result.final_output,
        "tool_calls": tool_calls,
        "action_calls": [],
    }


def run_billing_agent_threadsafe(
    customer_message: str,
    model: str = "gpt-4o-mini",
    instructions: str | None = None,
) -> dict:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_billing_agent, customer_message, model, instructions)
        return future.result()


def _seed_account_record(case: BillingCase | EscalationCase) -> BillingAccountRecord:
    if case["category"] == "billing":
        return BillingAccountRecord.from_billing_source(case["source_data"])
    return BillingAccountRecord.from_escalation_source(case["source_data"])


def _seed_invoice_records(case: BillingCase) -> tuple[str, list[InvoiceRecord]]:
    source_data = case["source_data"]
    invoice_sources = sorted(
        source_data["invoices"],
        key=lambda invoice: (invoice["issued_on"], invoice["invoice_id"]),
        reverse=True,
    )
    return source_data["customer_id"], [InvoiceRecord.from_source(invoice) for invoice in invoice_sources]


def hydrate_backend_from_dataset(dataset: list[SupportCase]) -> BackendSnapshot:
    BILLING_ACCOUNT_DB.clear()
    ACCOUNT_INVOICE_DB.clear()

    for case in dataset:
        if case["category"] not in {"billing", "escalation"}:
            continue

        hydration_case = cast(BillingCase | EscalationCase, case)
        account_id = hydration_case["source_data"]["customer_id"]
        BILLING_ACCOUNT_DB.setdefault(account_id, _seed_account_record(hydration_case))

        if hydration_case["category"] != "billing":
            continue

        invoice_account_id, invoice_records = _seed_invoice_records(hydration_case)
        ACCOUNT_INVOICE_DB.setdefault(invoice_account_id, invoice_records)

    return snapshot_backend()
