import hashlib
import json
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
import redis.asyncio as aioredis

from backend.db.pool import get_pool
from backend.config import get_settings
from backend.resilience.backpressure import check_queue_depth

router = APIRouter()
UPLOAD_DIR = Path("/tmp/neuroflow_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

ALLOWED_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "image/jpeg": "image",
    "image/png": "image",
    "image/webp": "image",
    "text/csv": "csv",
}


class URLIngestRequest(BaseModel):
    url: str
    pipeline_id: str | None = None


async def _get_redis():
    settings = get_settings()
    return aioredis.from_url(settings.redis_url)


@router.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    settings = get_settings()

    # Check backpressure
    r = await _get_redis()
    await check_queue_depth(r, settings)

    # Validate content type
    source_type = ALLOWED_TYPES.get(file.content_type)
    if not source_type:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    # Read and size-check
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 100MB limit")

    # Deduplication
    content_hash = hashlib.sha256(data).hexdigest()
    pool = await get_pool()

    existing = await pool.fetchrow(
        "SELECT id FROM documents WHERE content_hash=$1", content_hash
    )
    if existing:
        return {"document_id": str(existing["id"]), "status": "complete", "duplicate": True}

    # Save file
    doc_id = uuid.uuid4()
    file_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"
    file_path.write_bytes(data)

    # Create DB record
    await pool.execute(
        """INSERT INTO documents (id, filename, source_type, content_hash, status)
           VALUES ($1, $2, $3, $4, 'queued')""",
        doc_id, file.filename, source_type, content_hash,
    )

    # Enqueue job
    job = json.dumps({
        "document_id": str(doc_id),
        "file_path": str(file_path),
        "source_type": source_type,
    })
    await r.lpush("queue:ingest", job)
    await r.aclose()

    return {"document_id": str(doc_id), "status": "queued", "duplicate": False}


@router.post("/ingest/url")
async def ingest_url(body: URLIngestRequest):
    settings = get_settings()
    r = await _get_redis()
    await check_queue_depth(r, settings)

    url_hash = hashlib.sha256(body.url.encode()).hexdigest()
    pool = await get_pool()

    existing = await pool.fetchrow("SELECT id FROM documents WHERE content_hash=$1", url_hash)
    if existing:
        return {"document_id": str(existing["id"]), "status": "complete", "duplicate": True}

    doc_id = uuid.uuid4()
    await pool.execute(
        """INSERT INTO documents (id, filename, source_type, content_hash, status)
           VALUES ($1, $2, 'url', $3, 'queued')""",
        doc_id, body.url, url_hash,
    )

    job = json.dumps({"document_id": str(doc_id), "url": body.url, "source_type": "url"})
    await r.lpush("queue:ingest", job)
    await r.aclose()

    return {"document_id": str(doc_id), "status": "queued", "duplicate": False}


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id, filename, source_type, status, chunk_count, created_at FROM documents WHERE id=$1",
        uuid.UUID(document_id),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return dict(row)
