"""SDK quickstart example — runs against a live NeuroFlow instance."""
import asyncio
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from sdk.neuroflow import NeuroFlowClient


async def main():
    base_url = os.getenv("NEUROFLOW_URL", "http://localhost:8000")
    api_key = os.getenv("NEUROFLOW_API_KEY", "")

    client = NeuroFlowClient(base_url=base_url, api_key=api_key)

    # List pipelines
    pipelines = await client.list_pipelines()
    print(f"Found {len(pipelines)} pipelines")

    if not pipelines:
        print("No pipelines found. Create one first via POST /api/v1/pipelines")
        return

    pipeline_id = pipelines[0]["id"]
    print(f"Using pipeline: {pipeline_id}")

    # Stream a query
    print("\nQuerying with streaming...\n")
    async for token in await client.query(
        "What is retrieval augmented generation?",
        pipeline_id=pipeline_id,
        stream=True,
    ):
        print(token, end="", flush=True)
    print("\n\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
