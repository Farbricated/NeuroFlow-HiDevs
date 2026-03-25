import json
from backend.providers.base import ChatMessage, RoutingCriteria
from backend.providers.client import get_client


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x ** 2 for x in a) ** 0.5
    nb = sum(x ** 2 for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


async def evaluate_answer_relevance(query: str, answer: str) -> float:
    if not answer.strip():
        return 0.0

    client = get_client()
    criteria = RoutingCriteria(task_type="evaluation")

    gen_prompt = f"""Given this answer, generate 4 questions that this answer could be responding to.
Answer: {answer}
Return ONLY a JSON array of question strings. No other text."""

    result = await client.chat(
        [ChatMessage(role="user", content=gen_prompt)],
        routing_criteria=criteria,
        max_tokens=300,
    )

    try:
        text = result.content.strip()
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        generated_questions = json.loads(text)
    except Exception:
        return 0.5

    if not generated_questions:
        return 0.5

    all_texts = [query] + generated_questions
    embeddings = await client.embed(all_texts)

    query_emb = embeddings[0]
    q_embs = embeddings[1:]

    scores = [_cosine(query_emb, q) for q in q_embs]
    return sum(scores) / len(scores)
