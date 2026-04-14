import json
from pathlib import Path
from typing import Literal, TypedDict, cast

_DATASET_PATH = Path(__file__).with_name("dataset.json")


class BillingSourceData(TypedDict):
    customer_id: str
    account_name: str
    invoice_id: str
    plan_name: str
    last_charge_amount: float
    duplicate_charge: bool
    credit_eligible: bool
    billing_status: str
    notes: str


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
    difficulty: str
    is_edge_case: bool
    user_input: str
    source_data: dict[str, object]
    expected_output: str
    missing_info_required: bool
    workflow_expectation: str
    action_expected: bool
    action_type: str | None


class BillingCase(BaseCase):
    category: Literal["billing"]
    source_data: BillingSourceData
    action_type: str | None


class EscalationCase(BaseCase):
    category: Literal["escalation"]
    source_data: EscalationSourceData
    action_type: str | None


SupportCase = BaseCase


DATASET = cast(list[SupportCase], json.loads(_DATASET_PATH.read_text()))
