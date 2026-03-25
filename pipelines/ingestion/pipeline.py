import hashlib
import json
import uuid
from pathlib import Path
from opentelemetry import trace

from pipelines.ingestion.extractors.pdf_extractor import extract_pdf
from pipelines.ingestion.extractors.docx_extractor import extract_docx
from pipelines.ingestion.extractors.image_extractor import extract_image
from pipelines.ingestion.extractors.csv_extractor import extract_csv
from pipelines.ingestion.extractors.url_extractor import extract_url
from pipelines.ingestion.chunker import chunk_pages
from backend.db.pool import get_pool
from backend.providers.client import get_client

tracer = trace.get_tracer(__name__)

EXTRACTORS = {
    "pdf": extract_pdf,
    "docx": extract_docx,
    "image": extract_image,
    "csv": extract_csv,
    "url": extract_url,
}


def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def process_document(job: dict) -> None:
    document_id = job["document_id"]
    source_type = job["source_type"]
    file_path = job.get("file_path")
    url = job.get("url")

    pool = await get_pool()
    client = get_client()

    with tracer.start_as_current_span("ingestion.process") as span:
        span.set_attribute("document_id", document_id)
        span.set_attribute("source_type", source_type)

        try:
            # Mark as processing
            await pool.execute(
                "UPDATE documents SET status='processing' WHERE id=$1",
                uuid.UUID(document_id),
            )

            # Extract pages
            extractor = EXTRACTORS.get(source_type)
            if not extractor:
                raise ValueError(f"No extractor for source_type: {source_type}")

            if source_type == "url":
                pages = await extractor(url)
            else:
                pages = await extractor(file_path)

            span.set_attribute("page_count", len(pages))

            # Chunk
            chunks = await chunk_pages(pages)
            span.set_attribute("chunk_count", len(chunks))

            # Embed all chunks in one batch call
            texts = [c.content for c in chunks]
            embeddings = await client.embed(texts)
            span.set_attribute("embedding_calls", len(texts) // 100 + 1)

            # Insert chunks into DB
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        await conn.execute(
                            """
                            INSERT INTO chunks
                              (document_id, content, embedding, chunk_index, token_count, metadata)
                            VALUES ($1, $2, $3::vector, $4, $5, $6)
                            """,
                            uuid.UUID(document_id),
                            chunk.content,
                            json.dumps(embedding),
                            chunk.chunk_index,
                            chunk.token_count,
                            json.dumps(chunk.metadata),
                        )

                    await conn.execute(
                        "UPDATE documents SET status='complete', chunk_count=$1 WHERE id=$2",
                        len(chunks),
                        uuid.UUID(document_id),
                    )

        except Exception as e:
            await pool.execute(
                "UPDATE documents SET status='failed' WHERE id=$1",
                uuid.UUID(document_id),
            )
            span.record_exception(e)
            raise
