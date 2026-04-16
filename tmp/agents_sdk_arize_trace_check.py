"""Scratch harness for comparing Arize tracing behavior across raw Responses API and Agents SDK runs."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from enum import Enum
import argparse
import os
from pathlib import Path

from agents import set_default_openai_key
from arize.otel import register
from dotenv import load_dotenv
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

from workshop_helpers.backend import hydrate_backend_from_dataset, run_billing_agent_threadsafe
from workshop_helpers.data import DATASET


class Mode(str, Enum):
    OPENAI_ONLY = "openai_only"
    AGENTS_ONLY = "agents_only"
    BOTH_GLOBAL = "both_global"
    BOTH_SCOPED = "both_scoped"


PROMPT_ROUTER = (
    "You are a support routing classifier. "
    "Classify each user question into exactly one category from this list: "
    "permissions, review_workflow, billing, escalation. "
    "Return strict JSON with one key: category."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in Mode],
        default=Mode.BOTH_SCOPED.value,
        help="Tracing setup to test.",
    )
    parser.add_argument(
        "--skip-router",
        action="store_true",
        help="Skip the plain OpenAI router call and only run the billing agent.",
    )
    parser.add_argument(
        "--project-name",
        default="",
        help="Optional fixed Arize project name to export to.",
    )
    return parser.parse_args()


def build_tracer_provider(project_name: str):
    return register(
        space_id=os.environ["ARIZE_SPACE_ID"],
        api_key=os.environ["ARIZE_API_KEY"],
        project_name=project_name,
        batch=False,
        verbose=False,
    )


def instrument_for_mode(mode: Mode, tracer_provider):
    openai_instr = OpenAIInstrumentor()
    agents_instr = OpenAIAgentsInstrumentor()

    if mode in {Mode.OPENAI_ONLY, Mode.BOTH_GLOBAL, Mode.BOTH_SCOPED}:
        openai_instr.instrument(tracer_provider=tracer_provider)
    if mode in {Mode.AGENTS_ONLY, Mode.BOTH_GLOBAL, Mode.BOTH_SCOPED}:
        agents_instr.instrument(tracer_provider=tracer_provider)

    return openai_instr, agents_instr


@contextmanager
def maybe_scope_openai_off(mode: Mode, openai_instr: OpenAIInstrumentor, tracer_provider):
    if mode is not Mode.BOTH_SCOPED:
        yield
        return

    openai_instr.uninstrument()
    try:
        yield
    finally:
        openai_instr.instrument(tracer_provider=tracer_provider)


def run_router_call(client: OpenAI, case: dict) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=PROMPT_ROUTER,
        input=case["user_input"],
        temperature=0,
        max_output_tokens=80,
    )
    return response.output_text


def run_billing_case(case: dict) -> dict:
    account_id = case["source_data"]["customer_id"]
    instructions = (
        "You are a billing support specialist for an AI support copilot. "
        f"The authenticated billing account ID for this session is: {account_id}. "
        "Draft a reply only; do not claim you applied credits, updated invoices, reversed charges, or escalated a case. "
        "Always call get_billing_policy first to retrieve the applicable billing policy before any other billing tool. "
        "After retrieving the policy, use get_billing_account and list_invoices to verify facts before replying. "
        "If the account appears eligible for a credit or manual billing review, explain the next step and ask for confirmation before any action is taken."
    )
    return run_billing_agent_threadsafe(
        customer_message=case["user_input"],
        instructions=instructions,
    )


def main() -> None:
    args = parse_args()
    mode = Mode(args.mode)

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
    set_default_openai_key(os.environ["OPENAI_API_KEY"], use_for_tracing=False)

    project_name = args.project_name or f"Workiva-Temp-{mode.value}-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    tracer_provider = build_tracer_provider(project_name)

    client = OpenAI()
    openai_instr, _ = instrument_for_mode(mode, tracer_provider)

    hydrate_backend_from_dataset(DATASET)
    router_case = next(item for item in DATASET if item["scenario_id"] == "CP_001")
    billing_case = next(item for item in DATASET if item["scenario_id"] == "CB_004")

    print(f"Arize project: {project_name}")
    print(f"Mode: {mode.value}")
    print()

    if not args.skip_router:
        router_output = run_router_call(client, router_case)
        print(f"Router output: {router_output}")

    with maybe_scope_openai_off(mode, openai_instr, tracer_provider):
        billing_result = run_billing_case(billing_case)

    print(f"Billing final output: {billing_result['output']}")
    print(f"Billing tool calls: {[call['name'] for call in billing_result['tool_calls']]}")
    print(f"Billing action calls: {[call['name'] for call in billing_result['action_calls']]}")
    print()

    print("Flushing traces to Arize...")
    print(f"force_flush={tracer_provider.force_flush()}")
    tracer_provider.shutdown()


if __name__ == "__main__":
    main()
