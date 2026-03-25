"""NeuroFlow Python SDK client."""
import asyncio
import json
import time
from pathlib import Path
from typing import AsyncGenerator, Optional
import httpx

from .models import Document, QueryResult, EvaluationResult, Citation


class NeuroFlowClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout

    async def _get(self, path: str, **kwargs) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self.base_url}{path}", headers=self._headers, **kwargs)
            resp.raise_for_status()
            return resp.json()

    async def _post(self, path: str, **kwargs) -> dict:
        for attempt in range(3):
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self.base_url}{path}", headers=self._headers, **kwargs)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
                    await asyncio.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp.json()
        raise RuntimeError("Rate limit exceeded after 3 retries")

    async def ingest_file(
        self, file_path: str | Path, pipeline_id: Optional[str] = None, wait: bool = True
    ) -> Document:
        path = Path(file_path)
        with open(path, "rb") as f:
            data = await self._post(
                "/api/v1/ingest",
                files={"file": (path.name, f, "application/octet-stream")},
            )
        doc = Document(
            document_id=data["document_id"],
            filename=path.name,
            status=data["status"],
            duplicate=data.get("duplicate", False),
        )
        if wait and doc.status not in ("complete", "failed"):
            doc = await self._wait_for_ingestion(doc.document_id)
        return doc

    async def ingest_url(self, url: str, pipeline_id: Optional[str] = None, wait: bool = True) -> Document:
        data = await self._post("/api/v1/ingest/url", json={"url": url})
        doc = Document(document_id=data["document_id"], filename=url, status=data["status"])
        if wait and doc.status not in ("complete", "failed"):
            doc = await self._wait_for_ingestion(doc.document_id)
        return doc

    async def _wait_for_ingestion(self, document_id: str, timeout: int = 120) -> Document:
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = await self._get(f"/api/v1/documents/{document_id}")
            if data["status"] in ("complete", "failed"):
                return Document(
                    document_id=document_id,
                    filename=data.get("filename", ""),
                    status=data["status"],
                    chunk_count=data.get("chunk_count"),
                )
            await asyncio.sleep(3)
        raise TimeoutError(f"Ingestion did not complete in {timeout}s")

    async def query(
        self,
        query: str,
        pipeline_id: str,
        stream: bool = False,
    ) -> "QueryResult | AsyncGenerator[str, None]":
        if stream:
            return self._stream_query(query, pipeline_id)
        data = await self._post("/api/v1/query", json={"query": query, "pipeline_id": pipeline_id, "stream": False})
        return QueryResult(
            run_id=data.get("run_id", ""),
            generation=data.get("generation", ""),
            citations=[Citation(**c) for c in data.get("citations", [])],
            latency_ms=data.get("latency_ms"),
        )

    async def _stream_query(self, query: str, pipeline_id: str) -> AsyncGenerator[str, None]:
        data = await self._post("/api/v1/query", json={"query": query, "pipeline_id": pipeline_id, "stream": True})
        run_id = data["run_id"]
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "GET",
                f"{self.base_url}/api/v1/query/{run_id}/stream",
                headers=self._headers,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        try:
                            event = json.loads(line[5:].strip())
                            if event.get("type") == "token":
                                yield event.get("delta", "")
                        except json.JSONDecodeError:
                            pass

    async def get_evaluation(self, run_id: str, wait: bool = True, timeout: int = 120) -> EvaluationResult:
        deadline = time.time() + timeout
        while True:
            try:
                items = await self._get("/api/v1/evaluations", params={"per_page": 1})
                for item in items.get("items", []):
                    if item["run_id"] == run_id:
                        return EvaluationResult(
                            run_id=run_id,
                            faithfulness=item["faithfulness"],
                            answer_relevance=item["answer_relevance"],
                            context_precision=item["context_precision"],
                            context_recall=item["context_recall"],
                            overall_score=item["overall_score"],
                        )
            except Exception:
                pass
            if not wait or time.time() > deadline:
                raise TimeoutError(f"Evaluation for run {run_id} not found in {timeout}s")
            await asyncio.sleep(5)

    async def list_pipelines(self) -> list[dict]:
        return await self._get("/api/v1/pipelines")

    async def create_pipeline(self, config: dict) -> dict:
        return await self._post("/api/v1/pipelines", json=config)
