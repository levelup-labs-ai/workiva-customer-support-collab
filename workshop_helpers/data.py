import json
from pathlib import Path
from typing import Literal, TypedDict, cast

_DATASET_PATH = Path(__file__).with_name("dataset.json")


class BillingLineItemSourceData(TypedDict):
    description: str
    amount: float
    quantity: float
    unit_price: float


class BillingInvoiceSourceData(TypedDict):
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
    line_items: list[BillingLineItemSourceData]


class BillingSourceData(TypedDict):
    customer_id: str
    account_name: str
    plan_name: str
    credit_eligible: bool
    billing_status: str
    account_context: dict[str, object]
    invoices: list[BillingInvoiceSourceData]


class EscalationSourceData(TypedDict):
    customer_id: str
    account_name: str
    account_tier: str
    risk_level: str
    deadline: str
    recent_contacts: int
    notes: str


class BaseCase(TypedDict):
    scenario_id: str
    category: str
    user_input: str
    source_data: dict[str, object]
    workflow_expectation: str


class BillingCase(BaseCase):
    category: Literal["billing"]
    source_data: BillingSourceData


class EscalationCase(BaseCase):
    category: Literal["escalation"]
    source_data: EscalationSourceData


SupportCase = BaseCase


DATASET = cast(list[SupportCase], json.loads(_DATASET_PATH.read_text()))
