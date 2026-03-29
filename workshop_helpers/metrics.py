from arize.experiments.evaluators.base import EvaluationResult, Evaluator

LABEL_SCORE = {"Good": 1.0, "Acceptable": 0.5, "Poor": 0.0}

TONE_JUDGE = (
    "You evaluate customer support responses for tone.\n\n"
    "GOOD: Warm, empathetic, and professional.\n"
    "ACCEPTABLE: Polite but generic or slightly flat.\n"
    "POOR: Cold, robotic, dismissive, or argumentative.\n\n"
    "Respond with exactly one word: Good, Acceptable, or Poor."
)

OUTCOME_JUDGE = (
    "Compare a support response to the ideal response for the case.\n\n"
    "Customer message:\n{user_input}\n\n"
    "Ideal response:\n{ideal}\n\n"
    "Actual response:\n{actual}\n\n"
    "GOOD: The actual response reaches the same core outcome as the ideal response.\n"
    "ACCEPTABLE: Mostly correct but missing important specifics or completeness.\n"
    "POOR: Wrong decision, misses the main point, or fails to address the user's need.\n\n"
    "Respond with exactly one word: Good, Acceptable, or Poor."
)

WORKFLOW_FIT_JUDGE = (
    "Evaluate whether the response matches the expected workflow behavior for this capability stage.\n\n"
    "Benchmark slice: {benchmark_slice}\n"
    "Expected workflow behavior: {expected_behavior}\n"
    "Customer message:\n{user_input}\n\n"
    "Response:\n{actual}\n\n"
    "Use this rubric:\n"
    "- prompt / ask_one_targeted_followup: good responses acknowledge limited information and ask one focused follow-up instead of inventing facts.\n"
    "- context / answer_with_specific_context_no_action_claim: good responses use the support snapshot to answer specifically, but do not pretend a backend action already happened.\n"
    "- tools / confirm_backend_action_taken: good responses confirm a concrete backend action or result when enough information exists; merely suggesting next steps is weaker.\n\n"
    "GOOD: Matches the expected workflow behavior for the slice.\n"
    "ACCEPTABLE: Partially matches but is missing discipline or specificity.\n"
    "POOR: Behaves like the wrong capability stage.\n\n"
    "Respond with exactly one word: Good, Acceptable, or Poor."
)


def _one_word_judge(client, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=5,
    )
    return response.choices[0].message.content.strip().capitalize()


def judge_tone(client, output: str) -> str:
    return _one_word_judge(client, TONE_JUDGE, f"Response:\n{output}")


def judge_outcome(client, output: str, case: dict) -> str:
    return _one_word_judge(
        client,
        "You are a strict evaluator of support response correctness.",
        OUTCOME_JUDGE.format(
            user_input=case.get("user_input", ""),
            ideal=case.get("expected_output", ""),
            actual=output,
        ),
    )


def judge_workflow_fit(client, output: str, case: dict) -> str:
    return _one_word_judge(
        client,
        "You are a strict evaluator of whether a response matches the intended system capability.",
        WORKFLOW_FIT_JUDGE.format(
            benchmark_slice=case.get("benchmark_slice", "unknown"),
            expected_behavior=case.get("expected_behavior", "unknown"),
            user_input=case.get("user_input", ""),
            actual=output,
        ),
    )


def composite_score(tone: str, outcome: str, workflow_fit: str) -> float:
    return round(
        LABEL_SCORE.get(tone, 0.0)
        + LABEL_SCORE.get(outcome, 0.0)
        + LABEL_SCORE.get(workflow_fit, 0.0),
        1,
    )


def score_single_response(client, output: str, case: dict) -> dict:
    tone = judge_tone(client, output)
    outcome = judge_outcome(client, output, case)
    workflow_fit = judge_workflow_fit(client, output, case)
    return {
        "tone": tone,
        "correct_outcome": outcome,
        "workflow_fit": workflow_fit,
        "total": composite_score(tone, outcome, workflow_fit),
    }


def compare_scores(client, outputs: dict, case: dict) -> list[dict]:
    rows = []
    for label, output in outputs.items():
        rows.append({"variant": label, **score_single_response(client, output, case)})
    return rows


class ToneQualityEvaluator(Evaluator):
    def __init__(self, client):
        self.client = client

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        label = judge_tone(self.client, output or "")
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation="AI judge")


class CorrectOutcomeEvaluator(Evaluator):
    def __init__(self, client, dataset_by_id: dict):
        self.client = client
        self.dataset_by_id = dataset_by_id

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        case = self.dataset_by_id.get(dataset_row.get("scenario_id"), {})
        label = judge_outcome(self.client, output or "", case)
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation="AI judge")


class WorkflowFitEvaluator(Evaluator):
    def __init__(self, client, dataset_by_id: dict):
        self.client = client
        self.dataset_by_id = dataset_by_id

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        case = self.dataset_by_id.get(dataset_row.get("scenario_id"), {})
        label = judge_workflow_fit(self.client, output or "", case)
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation="AI judge")


def build_evaluators(client, dataset_by_id: dict) -> list[Evaluator]:
    return [
        ToneQualityEvaluator(client),
        CorrectOutcomeEvaluator(client, dataset_by_id),
        WorkflowFitEvaluator(client, dataset_by_id),
    ]
