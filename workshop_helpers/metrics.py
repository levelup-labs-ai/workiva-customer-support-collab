import json
import re

from arize.experiments.evaluators.base import EvaluationResult, Evaluator

LABEL_SCORE = {"Good": 1.0, "Acceptable": 0.5, "Poor": 0.0}
VARIANT_DISPLAY = {
    "router": "router",
    "permissions": "permissions",
    "review_workflow": "review_workflow",
    "billing": "billing",
    "v2_routed": "v2_routed",
}


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
                "metadata": payload.get("metadata", {}),
            }
    except Exception:
        pass
    return {"response_text": output or "", "tool_calls": [], "action_calls": [], "metadata": {}}


def pack_response_payload(
    response_text: str,
    tool_calls: list | None = None,
    action_calls: list | None = None,
    metadata: dict | None = None,
) -> str:
    return json.dumps(
        {
            "response_text": response_text,
            "tool_calls": tool_calls or [],
            "action_calls": action_calls or [],
            "metadata": metadata or {},
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


def _judge_with_reasoning(client, system_prompt: str, user_prompt: str) -> tuple[str, str]:
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_output_tokens=140,
    )
    content = response.output_text.strip()
    return _parse_judge_response(content)


def judge_brand_voice(client, output: str, judge_prompts: dict) -> tuple[str, str]:
    payload = _parse_task_output(output)
    return _judge_with_reasoning(client, judge_prompts["brand_voice"], f"Response:\n{payload['response_text']}")


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


class RoutingAccuracyEvaluator(ExactMatchEvaluator):
    pass


class BrandVoiceEvaluator(Evaluator):
    def __init__(self, client, judge_prompts: dict):
        self.client = client
        self.judge_prompts = judge_prompts

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        label, reasoning = judge_brand_voice(self.client, output or "", judge_prompts=self.judge_prompts)
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation=reasoning)


def build_evaluators(
    client,
    dataset_by_id: dict,
    variant_name: str,
    variant_behavior: str,
    judge_prompts: dict,
) -> list[Evaluator]:
    if variant_name == "router":
        return [RoutingAccuracyEvaluator(expected_field="category")]

    return [BrandVoiceEvaluator(client, judge_prompts=judge_prompts)]
