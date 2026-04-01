import json
import re

from arize.experiments.evaluators.base import EvaluationResult, Evaluator

LABEL_SCORE = {"Good": 1.0, "Acceptable": 0.5, "Poor": 0.0}
VARIANT_DISPLAY = {"router": "router", "v1": "v1", "v2": "v2", "v3": "v3"}

TONE_JUDGE = (
    "You evaluate customer support responses for tone.\n\n"
    "Return exactly two lines in this format:\n"
    "LABEL: <Good|Acceptable|Poor>\n"
    "REASONING: <one sentence>\n\n"
    "GOOD: Warm, empathetic, and professional.\n"
    "ACCEPTABLE: Polite but generic or slightly flat.\n"
    "POOR: Cold, robotic, dismissive, or argumentative."
)

OUTCOME_JUDGE = (
    "Evaluate whether the response reaches the right substantive outcome for the case.\n\n"
    "Customer message:\n{user_input}\n\n"
    "Expected output:\n{ideal}\n\n"
    "Actual response:\n{actual}\n\n"
    "Return exactly two lines in this format:\n"
    "LABEL: <Good|Acceptable|Poor>\n"
    "REASONING: <one sentence>\n\n"
    "GOOD: Reaches the same core outcome as the expected output.\n"
    "ACCEPTABLE: Mostly correct but missing important specifics or completeness.\n"
    "POOR: Wrong outcome, misses the main point, or fails to address the user need."
)

WORKFLOW_JUDGE = (
    "Evaluate whether the response matches the expected workflow behavior for this system variant.\n\n"
    "System variant: {variant_name}\n"
    "Expected variant behavior: {variant_behavior}\n"
    "Variant-specific expectation for this case: {variant_expectation}\n"
    "Case workflow expectation: {workflow_expectation}\n"
    "Missing info required: {missing_info_required}\n"
    "Action expected: {action_expected}\n"
    "Action type: {action_type}\n\n"
    "Customer message:\n{user_input}\n\n"
    "Actual response:\n{actual}\n\n"
    "Return exactly two lines in this format:\n"
    "LABEL: <Good|Acceptable|Poor>\n"
    "REASONING: <one sentence>\n\n"
    "GOOD: Clearly matches the expected workflow behavior for this variant.\n"
    "ACCEPTABLE: Partially matches but is missing specificity or discipline.\n"
    "POOR: Does not match the expected workflow behavior for this variant."
)


def _parse_judge_response(text: str) -> tuple[str, str]:
    label_match = re.search(r"LABEL:\s*(Good|Acceptable|Poor)", text, re.IGNORECASE)
    reasoning_match = re.search(r"REASONING:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    label = label_match.group(1).capitalize() if label_match else "Poor"
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()
    return label, reasoning


def _parse_task_output(output: str) -> dict:
    try:
        payload = json.loads(output)
        if isinstance(payload, dict) and "response_text" in payload:
            return {
                "response_text": payload.get("response_text", ""),
                "tool_calls": payload.get("tool_calls", []),
                "action_calls": payload.get("action_calls", []),
            }
    except Exception:
        pass
    return {"response_text": output or "", "tool_calls": [], "action_calls": []}


def pack_response_payload(response_text: str, tool_calls: list | None = None, action_calls: list | None = None) -> str:
    return json.dumps(
        {
            "response_text": response_text,
            "tool_calls": tool_calls or [],
            "action_calls": action_calls or [],
        }
    )


def normalize_text_label(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def exact_match_result(actual: str | None, expected: str | None) -> tuple[str, str]:
    normalized_actual = normalize_text_label(actual)
    normalized_expected = normalize_text_label(expected)
    if normalized_actual == normalized_expected:
        return "Good", f"Exact match on normalized label `{expected}`."
    return "Poor", f"Predicted `{actual or ''}` but expected `{expected or ''}`."


def _variant_expectation(case: dict, variant_name: str) -> str:
    missing_info_required = case.get("missing_info_required", False)
    action_expected = case.get("action_expected", False)
    action_type = case.get("action_type", "none")
    workflow_expectation = case.get("workflow_expectation", "unknown")

    if variant_name == "v1":
        if workflow_expectation == "ask_followup":
            return "Ask the minimal necessary follow-up question or questions, and avoid bluffing."
        if action_expected:
            return (
                "Do not claim the action was completed. Because this variant only sees the raw customer message, "
                "a good response may ask the minimal necessary follow-up question or questions, or give a cautious, honest next step "
                "without pretending backend execution happened."
            )
        return (
            "Give a helpful, honest best-effort answer using only the customer message. "
            "Do not pretend to have backend access, and do not penalize reasonable minimal clarifying questions."
        )

    if variant_name == "v2":
        if action_expected:
            return (
                f"Use the provided context to explain the right next step for `{action_type}`, "
                "but do not claim the backend action already happened."
            )
        if missing_info_required:
            return "Ask only for the missing information that is still needed after using the provided context."
        return "Use the provided context specifically and give the user a clear next step."

    if action_expected:
        return f"Use tools to complete `{action_type}` and confirm the action result in the final response."
    if missing_info_required:
        return "Use tools to verify what you can first, then ask only if something is still missing."
    return "Use tools when helpful and answer the user directly with verified information."


def _judge_with_reasoning(client, system_prompt: str, user_prompt: str) -> tuple[str, str]:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=120,
    )
    content = response.choices[0].message.content.strip()
    return _parse_judge_response(content)


def judge_tone(client, output: str) -> tuple[str, str]:
    payload = _parse_task_output(output)
    return _judge_with_reasoning(client, TONE_JUDGE, f"Response:\n{payload['response_text']}")


def judge_outcome(client, output: str, case: dict) -> tuple[str, str]:
    payload = _parse_task_output(output)
    return _judge_with_reasoning(
        client,
        "You are a strict evaluator of support response correctness.",
        OUTCOME_JUDGE.format(
            user_input=case.get("user_input", ""),
            ideal=case.get("expected_output", ""),
            actual=payload["response_text"],
        ),
    )


def judge_workflow_fit(client, output: str, case: dict, variant_name: str, variant_behavior: str) -> tuple[str, str]:
    payload = _parse_task_output(output)
    tool_calls = payload["tool_calls"]
    action_calls = payload["action_calls"]
    tool_summary = "; ".join(
        (
            f"{call.get('name', 'unknown')}({call.get('arguments', '')})"
            if isinstance(call, dict)
            else str(call)
        )
        for call in tool_calls
    ) or "none"
    action_summary = "; ".join(
        (
            f"{call.get('name', 'unknown')}({call.get('arguments', '')})"
            if isinstance(call, dict)
            else str(call)
        )
        for call in action_calls
    ) or "none"
    return _judge_with_reasoning(
        client,
        "You are a strict evaluator of support workflow behavior.",
        WORKFLOW_JUDGE.format(
            variant_name=variant_name,
            variant_behavior=variant_behavior,
            variant_expectation=_variant_expectation(case, variant_name),
            workflow_expectation=case.get("workflow_expectation", "unknown"),
            missing_info_required=case.get("missing_info_required", False),
            action_expected=case.get("action_expected", False),
            action_type=case.get("action_type", "none"),
            user_input=case.get("user_input", ""),
            actual=payload["response_text"],
        )
        + f"\n\nRecorded tool calls: {tool_summary}\nRecorded action calls: {action_summary}",
    )


def _action_match_score(case: dict, output: str) -> tuple[str, str] | None:
    if not case.get("action_expected"):
        return None
    payload = _parse_task_output(output)
    expected_action = case.get("action_type")
    action_calls = payload["action_calls"]
    for call in action_calls:
        call_name = call.get("name") if isinstance(call, dict) else str(call)
        if expected_action and call_name == expected_action:
            arguments = call.get("arguments", "") if isinstance(call, dict) else ""
            return "Good", f"Recorded action tool `{expected_action}` was called with arguments: {arguments}"
    if action_calls:
        called_names = ", ".join(
            call.get("name", str(call)) if isinstance(call, dict) else str(call) for call in action_calls
        )
        return "Acceptable", (
            f"Recorded action tool calls were `{called_names}`, but none matched expected action `{expected_action}`."
        )
    return "Poor", f"No recorded action tool call matched expected action `{expected_action}`."


def composite_score(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return round(sum(scores), 1)


def score_routing_response(output: str, expected_category: str) -> dict:
    payload = _parse_task_output(output)
    label, reasoning = exact_match_result(payload["response_text"], expected_category)
    return {
        "exact_match": label,
        "exact_match_reasoning": reasoning,
        "predicted_category": payload["response_text"],
        "expected_category": expected_category,
        "total": LABEL_SCORE.get(label, 0.0),
    }


def score_single_response(client, output: str, case: dict, variant_name: str, variant_behavior: str) -> dict:
    tone_label, tone_reasoning = judge_tone(client, output)
    workflow_label, workflow_reasoning = judge_workflow_fit(
        client,
        output,
        case,
        variant_name=variant_name,
        variant_behavior=variant_behavior,
    )

    row = {
        "tone": tone_label,
        "tone_reasoning": tone_reasoning,
        "workflow_fit": workflow_label,
        "workflow_reasoning": workflow_reasoning,
        "correct_outcome": "N/A" if variant_name == "v1" else "",
        "outcome_reasoning": "" if variant_name == "v1" else "",
    }

    scores = [LABEL_SCORE.get(tone_label, 0.0), LABEL_SCORE.get(workflow_label, 0.0)]

    if variant_name != "v1":
        outcome_label, outcome_reasoning = judge_outcome(client, output, case)
        row["correct_outcome"] = outcome_label
        row["outcome_reasoning"] = outcome_reasoning
        scores.append(LABEL_SCORE.get(outcome_label, 0.0))
    action_check = _action_match_score(case, output) if variant_name == "v3" else None
    if action_check:
        row["action_execution"] = action_check[0]
        row["action_reasoning"] = action_check[1]
    else:
        row["action_execution"] = ""
        row["action_reasoning"] = ""

    row["total"] = composite_score(scores)
    return row


def compare_scores(client, outputs: dict, case: dict, variant_behaviors: dict | None = None) -> list[dict]:
    variant_behaviors = variant_behaviors or {}
    rows = []
    for label, output in outputs.items():
        rows.append(
            {
                "variant": label,
                **score_single_response(
                    client,
                    output,
                    case,
                    variant_name=label,
                    variant_behavior=variant_behaviors.get(label, label),
                ),
            }
        )
    return rows


class ExactMatchEvaluator(Evaluator):
    def __init__(self, expected_field: str, output_field: str = "response_text"):
        self.expected_field = expected_field
        self.output_field = output_field

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        payload = _parse_task_output(output or "")
        actual = payload.get(self.output_field, "")
        expected = dataset_row.get(self.expected_field, "")
        label, reasoning = exact_match_result(actual, expected)
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation=reasoning)


class ToneQualityEvaluator(Evaluator):
    def __init__(self, client):
        self.client = client

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        label, reasoning = judge_tone(self.client, output or "")
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation=reasoning)


class CorrectOutcomeEvaluator(Evaluator):
    def __init__(self, client, dataset_by_id: dict):
        self.client = client
        self.dataset_by_id = dataset_by_id

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        case = self.dataset_by_id.get(dataset_row.get("scenario_id"), {})
        label, reasoning = judge_outcome(self.client, output or "", case)
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation=reasoning)


class WorkflowFitEvaluator(Evaluator):
    def __init__(self, client, dataset_by_id: dict, variant_name: str, variant_behavior: str):
        self.client = client
        self.dataset_by_id = dataset_by_id
        self.variant_name = variant_name
        self.variant_behavior = variant_behavior

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        case = self.dataset_by_id.get(dataset_row.get("scenario_id"), {})
        label, reasoning = judge_workflow_fit(
            self.client,
            output or "",
            case,
            variant_name=self.variant_name,
            variant_behavior=self.variant_behavior,
        )
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation=reasoning)


class ActionExecutionEvaluator(Evaluator):
    def __init__(self, dataset_by_id: dict):
        self.dataset_by_id = dataset_by_id

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        case = self.dataset_by_id.get(dataset_row.get("scenario_id"), {})
        result = _action_match_score(case, output or "")
        if result is None:
            return EvaluationResult(score=1.0, label="Good", explanation="No action expected for this case.")
        label, reasoning = result
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation=reasoning)


def build_evaluators(client, dataset_by_id: dict, variant_name: str, variant_behavior: str) -> list[Evaluator]:
    if variant_name == "router":
        return [ExactMatchEvaluator(expected_field="category")]

    evaluators: list[Evaluator] = [
        ToneQualityEvaluator(client),
        WorkflowFitEvaluator(client, dataset_by_id, variant_name=variant_name, variant_behavior=variant_behavior),
    ]
    if variant_name != "v1":
        evaluators.insert(1, CorrectOutcomeEvaluator(client, dataset_by_id))
    if variant_name == "v3":
        evaluators.append(ActionExecutionEvaluator(dataset_by_id))
    return evaluators
