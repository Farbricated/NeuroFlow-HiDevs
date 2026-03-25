import re
import json
import uuid
import tiktoken
from pathlib import Path
from backend.db.pool import get_pool

enc = tiktoken.get_encoding("cl100k_base")
PII_PATTERNS = [
    re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'),  # email
    re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),                  # phone
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),                               # SSN
]


def _has_pii(text: str) -> bool:
    return any(p.search(text) for p in PII_PATTERNS)


def _count_tokens(text: str) -> int:
    return len(enc.encode(text))


async def extract_training_pairs(job_id: str, min_quality: float = 0.82) -> list[dict]:
    pool = await get_pool()

    rows = await pool.fetch(
        """SELECT tp.id, tp.run_id, tp.system_prompt, tp.user_message,
                  tp.assistant_message, tp.quality_score
           FROM training_pairs tp
           JOIN pipeline_runs pr ON pr.id = tp.run_id
           WHERE tp.quality_score >= $1
             AND tp.included_in_job IS NULL
             AND (pr.id NOT IN (
                   SELECT e.run_id FROM evaluations e WHERE e.user_rating <= 2
             ))
           ORDER BY tp.quality_score DESC""",
        min_quality,
    )

    valid_pairs = []
    for row in rows:
        user_msg = row["user_message"]
        asst_msg = row["assistant_message"]
        system = row["system_prompt"] or "You are a precise research assistant."

        # Validate token counts
        asst_tokens = _count_tokens(asst_msg)
        if not (50 <= asst_tokens <= 2000):
            continue

        # Must contain at least one citation
        if not re.search(r'\[Source \d+\]', asst_msg):
            continue

        # PII check
        if _has_pii(user_msg) or _has_pii(asst_msg):
            continue

        valid_pairs.append({
            "id": str(row["id"]),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst_msg},
            ],
            "quality_score": row["quality_score"],
        })

    if not valid_pairs:
        return []

    # Write JSONL
    output_path = Path(f"training_data/{job_id}.jsonl")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        for pair in valid_pairs:
            f.write(json.dumps({"messages": pair["messages"]}) + "\n")

    # Mark pairs as included
    pair_ids = [uuid.UUID(p["id"]) for p in valid_pairs]
    await pool.execute(
        "UPDATE training_pairs SET included_in_job=$1 WHERE id = ANY($2)",
        uuid.UUID(job_id), pair_ids,
    )

    return valid_pairs


async def preview_training_pairs(min_quality: float = 0.82, limit: int = 5) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT tp.user_message, tp.assistant_message, tp.quality_score
           FROM training_pairs tp
           WHERE tp.quality_score >= $1 AND tp.included_in_job IS NULL
           ORDER BY tp.quality_score DESC
           LIMIT $2""",
        min_quality, limit,
    )
    return [
        {
            "user_message": r["user_message"][:200] + "...",
            "assistant_preview": r["assistant_message"][:200] + "...",
            "quality_score": r["quality_score"],
        }
        for r in rows
    ]
