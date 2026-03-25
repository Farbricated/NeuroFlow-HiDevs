from backend.providers.base import ChatMessage, RoutingCriteria
from backend.providers.client import get_client


async def evaluate_context_precision(query: str, chunks: list[str], answer: str) -> float:
    if not chunks:
        return 0.0

    client = get_client()
    criteria = RoutingCriteria(task_type="evaluation")

    useful = []
    for chunk in chunks:
        prompt = f"""Was this passage useful in generating the answer to the query?
Query: {query}
Passage: {chunk[:500]}
Answer: {answer[:500]}
Respond with ONLY: yes or no"""

        result = await client.chat(
            [ChatMessage(role="user", content=prompt)],
            routing_criteria=criteria,
            max_tokens=5,
        )
        is_useful = result.content.strip().lower().startswith("yes")
        useful.append(is_useful)

    # Rank-weighted precision: Σ useful[i] * (1/i) / Σ (1/i)
    numerator = sum((1.0 / (i + 1)) for i, u in enumerate(useful) if u)
    denominator = sum(1.0 / (i + 1) for i in range(len(useful)))

    return numerator / denominator if denominator > 0 else 0.0
