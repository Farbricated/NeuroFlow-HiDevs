import json
import asyncio
from dataclasses import dataclass, field
from backend.providers.base import ChatMessage, RoutingCriteria
from backend.providers.client import get_client


@dataclass
class ProcessedQuery:
    original: str
    expansions: list[str] = field(default_factory=list)
    metadata_filters: dict = field(default_factory=dict)
    query_type: str = "factual"  # factual | analytical | comparative | procedural


async def process_query(query: str) -> ProcessedQuery:
    client = get_client()

    expansion_prompt = f"""Given this query: "{query}"
Generate 2-3 alternative phrasings that mean the same thing.
Also classify the query type as one of: factual, analytical, comparative, procedural.
Also extract any metadata filters (e.g. year, topic, author) as a JSON object.

Respond ONLY with JSON in this exact format:
{{
  "expansions": ["alt phrasing 1", "alt phrasing 2"],
  "query_type": "factual",
  "metadata_filters": {{}}
}}"""

    result = await client.chat(
        [ChatMessage(role="user", content=expansion_prompt)],
        routing_criteria=RoutingCriteria(task_type="classification"),
    )

    try:
        text = result.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
        return ProcessedQuery(
            original=query,
            expansions=parsed.get("expansions", []),
            metadata_filters=parsed.get("metadata_filters", {}),
            query_type=parsed.get("query_type", "factual"),
        )
    except Exception:
        return ProcessedQuery(original=query, expansions=[], metadata_filters={}, query_type="factual")
