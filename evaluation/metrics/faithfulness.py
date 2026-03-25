import json
from backend.providers.base import ChatMessage, RoutingCriteria
from backend.providers.client import get_client


async def evaluate_faithfulness(query: str, answer: str, context: str) -> float:
    if not answer.strip():
        return 0.0
    if not context.strip():
        return 0.0

    client = get_client()
    criteria = RoutingCriteria(task_type="evaluation")

    # Step 1: Extract claims
    extract_prompt = f"""Extract all factual claims from this answer as a JSON array of strings.
Answer: {answer}
Return ONLY a JSON array, e.g. ["claim 1", "claim 2"]. No other text."""

    claims_result = await client.chat(
        [ChatMessage(role="user", content=extract_prompt)],
        routing_criteria=criteria,
        max_tokens=500,
    )

    try:
        text = claims_result.content.strip()
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        claims = json.loads(text)
    except Exception:
        return 0.5  # Fallback if parsing fails

    if not claims:
        return 1.0

    # Step 2: Verify each claim against context
    supported = 0
    for claim in claims:
        verify_prompt = f"""Is this claim supported by the context?
Context: {context[:2000]}
Claim: {claim}
Answer with ONLY one word: yes, no, or partial."""

        result = await client.chat(
            [ChatMessage(role="user", content=verify_prompt)],
            routing_criteria=criteria,
            max_tokens=5,
        )
        verdict = result.content.strip().lower()
        if verdict == "yes":
            supported += 1
        elif verdict == "partial":
            supported += 0.5

    return supported / len(claims)
