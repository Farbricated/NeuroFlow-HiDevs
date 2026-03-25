BASE_SYSTEM = """You are a precise research assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer fully, say so explicitly.
For every factual claim, include a citation in the format [Source N].
Do not introduce information not present in the context."""

ADDONS = {
    "factual": "Provide a direct, concise answer. If multiple sources agree, cite all of them.",
    "analytical": "Analyze and synthesize across the provided sources. Identify agreements and contradictions.",
    "comparative": "Organize your response as a structured comparison. Use a table if appropriate.",
    "procedural": "Provide numbered steps. Each step must be cited.",
}


def build_prompt(query: str, context_text: str, query_type: str = "factual") -> list[dict]:
    addon = ADDONS.get(query_type, ADDONS["factual"])
    system = f"{BASE_SYSTEM}\n\n{addon}"

    user_content = f"<context>\n{context_text}\n</context>\n\nQuestion: {query}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
