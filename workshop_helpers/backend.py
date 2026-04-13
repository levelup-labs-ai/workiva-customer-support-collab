import asyncio
import concurrent.futures
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict, cast

from agents import Agent, Runner, function_tool

from workshop_helpers.data import BillingCase, BillingSourceData, EscalationCase, EscalationSourceData, SupportCase
from workshop_helpers.setup import suspend_openai_tracing_for_agents

_BILLING_REFERENCE_PATH = Path(__file__).with_name("billing_reference.json")


@dataclass(frozen=True)
class BillingAccountRecord:
    account_name: str
    plan_name: str
    billing_status: str
    credit_eligible: bool
    notes: str

    @classmethod
    def from_billing_source(cls, source_data: BillingSourceData) -> "BillingAccountRecord":
        return cls(
            account_name=source_data["account_name"],
            plan_name=source_data["plan_name"],
            billing_status=source_data["billing_status"],
            credit_eligible=source_data["credit_eligible"],
            notes=source_data["notes"],
        )

    @classmethod
    def from_escalation_source(cls, source_data: EscalationSourceData) -> "BillingAccountRecord":
        # Escalation cases seed the account lookup table even when there is no invoice metadata.
        return cls(
            account_name=source_data["account_name"],
            plan_name="Unknown",
            billing_status="active",
            credit_eligible=False,
            notes=source_data["notes"],
        )

    def to_tool_payload(self, account_id: str) -> dict:
        return {"account_id": account_id, **asdict(self)}


@dataclass(frozen=True)
class InvoiceRecord:
    account_id: str
    plan_name: str
    last_charge_amount: float
    duplicate_charge: bool
    credit_eligible: bool
    billing_status: str
    notes: str

    @classmethod
    def from_billing_source(cls, source_data: BillingSourceData) -> "InvoiceRecord":
        return cls(
            account_id=source_data["customer_id"],
            plan_name=source_data["plan_name"],
            last_charge_amount=source_data["last_charge_amount"],
            duplicate_charge=source_data["duplicate_charge"],
            credit_eligible=source_data["credit_eligible"],
            billing_status=source_data["billing_status"],
            notes=source_data["notes"],
        )

    def to_tool_payload(self, invoice_id: str) -> dict:
        return {"invoice_id": invoice_id, **asdict(self)}


class BackendSnapshot(TypedDict):
    billing_account_count: int
    invoice_count: int
    billing_reference_topics: int


BILLING_ACCOUNT_DB: dict[str, BillingAccountRecord] = {}
INVOICE_DB: dict[str, InvoiceRecord] = {}
BILLING_REFERENCE_DB: dict[str, dict] = (
    json.loads(_BILLING_REFERENCE_PATH.read_text()) if _BILLING_REFERENCE_PATH.exists() else {}
)

ACTION_RESULTS = {
    "apply_billing_credit": {
        "status": "credit_applied",
        "eta": "visible on the account within 1 business day",
    },
    "escalate_to_human": {
        "status": "escalated",
        "queue": "human_billing_specialist",
        "eta": "follow-up within 2 business hours",
    },
}

ACTION_TOOL_NAMES = list(ACTION_RESULTS.keys())


def snapshot_backend() -> BackendSnapshot:
    return {
        "billing_account_count": len(BILLING_ACCOUNT_DB),
        "invoice_count": len(INVOICE_DB),
        "billing_reference_topics": len(BILLING_REFERENCE_DB),
    }


@function_tool
def get_billing_account(account_id: str) -> dict:
    """Look up the billing account record for a customer or workspace."""
    account = BILLING_ACCOUNT_DB.get(account_id)
    if not account:
        return {"error": f"Billing account not found: {account_id}"}
    return account.to_tool_payload(account_id)


@function_tool
def get_invoice_details(invoice_id: str) -> dict:
    """Fetch the invoice metadata and issue flags for a billing case."""
    invoice = INVOICE_DB.get(invoice_id)
    if not invoice:
        return {"error": f"Invoice not found: {invoice_id}"}
    return invoice.to_tool_payload(invoice_id)


@function_tool
def read_billing_reference(topic: str) -> dict:
    """Read billing guidance from the local JSON reference file."""
    article = BILLING_REFERENCE_DB.get(topic)
    if not article:
        return {"error": f"No billing guidance found for topic: {topic}"}
    return {"topic": topic, **article}


def _perform_action(account_id: str, action: str, details: str) -> dict:
    result = dict(ACTION_RESULTS[action])
    result["action"] = action
    result["account_id"] = account_id
    result["details"] = details
    return result


@function_tool
def apply_billing_credit(account_id: str, details: str) -> dict:
    """Apply a billing credit or courtesy adjustment to the account."""
    return _perform_action(account_id, "apply_billing_credit", details)


@function_tool
def escalate_to_human(account_id: str, details: str) -> dict:
    """Escalate a case to a human billing specialist."""
    return _perform_action(account_id, "escalate_to_human", details)


TOOLS = [
    get_billing_account,
    get_invoice_details,
    read_billing_reference,
    apply_billing_credit,
    escalate_to_human,
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
    action_calls = []
    for item in result.new_items:
        raw = getattr(item, "raw_item", None)
        if not raw or not hasattr(raw, "name"):
            continue
        entry = {"name": raw.name}
        arguments = getattr(raw, "arguments", None)
        if arguments is not None:
            entry["arguments"] = arguments
        tool_calls.append(entry)
        if raw.name in ACTION_TOOL_NAMES:
            action_calls.append(entry)

    return {
        "output": result.final_output,
        "tool_calls": tool_calls,
        "action_calls": action_calls,
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


def _seed_invoice_record(case: BillingCase) -> tuple[str, InvoiceRecord]:
    source_data = case["source_data"]
    return source_data["invoice_id"], InvoiceRecord.from_billing_source(source_data)


def hydrate_backend_from_dataset(dataset: list[SupportCase]) -> BackendSnapshot:
    BILLING_ACCOUNT_DB.clear()
    INVOICE_DB.clear()

    for case in dataset:
        if case["category"] not in {"billing", "escalation"}:
            continue

        hydration_case = cast(BillingCase | EscalationCase, case)
        account_id = hydration_case["source_data"]["customer_id"]
        BILLING_ACCOUNT_DB.setdefault(account_id, _seed_account_record(hydration_case))

        if hydration_case["category"] != "billing":
            continue

        invoice_id, invoice_record = _seed_invoice_record(hydration_case)
        INVOICE_DB.setdefault(invoice_id, invoice_record)

    return snapshot_backend()
