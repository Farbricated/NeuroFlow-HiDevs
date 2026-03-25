# NeuroFlow Python SDK

## Installation
```bash
pip install ./sdk
```

## Quickstart (10 lines)
```python
import asyncio
from neuroflow import NeuroFlowClient

async def main():
    client = NeuroFlowClient(base_url="http://localhost:8000", api_key="<your-token>")

    # Ingest a document
    doc = await client.ingest_file("paper.pdf")
    print(f"Ingested: {doc.document_id} ({doc.chunk_count} chunks)")

    # Query with streaming
    async for token in await client.query("What is RAG?", pipeline_id="<id>", stream=True):
        print(token, end="", flush=True)

asyncio.run(main())
```
