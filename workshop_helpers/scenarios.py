import json
import re


def _messages(prompt: str, customer_message: str) -> list[dict]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": customer_message},
    ]


def _run_text_response(client, customer_message: str, prompt: str, max_tokens: int) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=_messages(prompt, customer_message),
        temperature=0.3,
        max_output_tokens=max_tokens,
    )
    return response.output_text.strip()


def run_raw_llm(client, customer_message: str, prompt: str, max_tokens: int = 220) -> str:
    return _run_text_response(client, customer_message, prompt, max_tokens)


def run_context_agent(
    client,
    customer_message: str,
    prompt: str,
    max_tokens: int = 220,
) -> str:
    return _run_text_response(client, customer_message, prompt, max_tokens)


def parse_router_raw_response(raw_response: str, categories: list[str]) -> dict:
    content = raw_response.strip()
    category = ""
    payload: dict = {"raw_response": content}

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

    if isinstance(parsed, dict):
        payload.update(parsed)
        category = str(parsed.get("category", "")).strip()
    else:
        category = content

    normalized_categories = {item.lower(): item for item in categories}
    normalized_category = category.lower()
    if normalized_category in normalized_categories:
        payload["category"] = normalized_categories[normalized_category]
    else:
        payload["category"] = category
        payload["fallback_reason"] = f"Invalid category returned: {category}"
    return payload


def run_router_raw(client, customer_message: str, prompt: str, categories: list[str]) -> dict:
    raw_response = run_raw_llm(client, customer_message, prompt, max_tokens=80)
    return parse_router_raw_response(raw_response, categories)
