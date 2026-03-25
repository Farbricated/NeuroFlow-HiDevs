import re
from backend.providers.base import ChatMessage, RoutingCriteria
from backend.providers.client import get_client


async def evaluate_context_recall(query: str, chunks: list[str], answer: str) -> float:
    if not answer.strip() or not chunks:
        return 0.0

    client = get_client()
    criteria = RoutingCriteria(task_type="evaluation")

    context = "\n\n".join(chunks)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
    if not sentences:
        return 0.0

    attributed = 0
    for sentence in sentences:
        prompt = f"""Can this sentence be attributed to the provided context?
Context: {context[:2000]}
Sentence: {sentence}
Respond with ONLY: yes or no"""

        result = await client.chat(
            [ChatMessage(role="user", content=prompt)],
            routing_criteria=criteria,
            max_tokens=5,
        )
        if result.content.strip().lower().startswith("yes"):
            attributed += 1

    return attributed / len(sentences)
