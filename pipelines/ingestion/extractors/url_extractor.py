import httpx
import trafilatura
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
from pipelines.ingestion.extractors.pdf_extractor import ExtractedPage

USER_AGENT = "NeuroFlowBot/1.0"


async def _check_robots(url: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(robots_url)
            content = resp.text
        rp = RobotFileParser()
        rp.parse(content.splitlines())
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True  # Allow if robots.txt can't be fetched


async def extract_url(url: str) -> list[ExtractedPage]:
    if not await _check_robots(url):
        raise PermissionError(f"Robots.txt disallows crawling: {url}")

    async with httpx.AsyncClient(timeout=15, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    content = trafilatura.extract(html, include_tables=True, include_links=False)
    if not content:
        raise ValueError(f"Could not extract content from {url}")

    metadata_raw = trafilatura.extract_metadata(html)
    metadata = {
        "url": url,
        "title": getattr(metadata_raw, "title", None),
        "author": getattr(metadata_raw, "author", None),
        "date": getattr(metadata_raw, "date", None),
    }

    return [ExtractedPage(
        page_number=1,
        content=content,
        content_type="text",
        metadata=metadata,
    )]
