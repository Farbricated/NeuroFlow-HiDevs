import re
import ipaddress
from urllib.parse import urlparse
import bleach
from fastapi import HTTPException

MAX_QUERY_LENGTH = 5000
MAX_NAME_LENGTH = 100
URL_PATTERN = re.compile(r"^https?://")

PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
]


def sanitize_text(text: str) -> str:
    return bleach.clean(text, tags=[], strip=True)


def validate_query(query: str) -> str:
    query = sanitize_text(query)
    if len(query) > MAX_QUERY_LENGTH:
        raise HTTPException(status_code=422, detail=f"Query exceeds {MAX_QUERY_LENGTH} characters")
    if not query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty")
    return query


def validate_name(name: str) -> str:
    name = sanitize_text(name)
    if len(name) > MAX_NAME_LENGTH:
        raise HTTPException(status_code=422, detail=f"Name exceeds {MAX_NAME_LENGTH} characters")
    return name


def validate_url(url: str) -> str:
    if not URL_PATTERN.match(url):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    # Block localhost
    if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        raise HTTPException(status_code=400, detail="Private/localhost URLs are not allowed (SSRF protection)")

    # Block private IP ranges
    try:
        addr = ipaddress.ip_address(hostname)
        for private_range in PRIVATE_RANGES:
            if addr in private_range:
                raise HTTPException(status_code=400, detail="Private IP ranges are not allowed (SSRF protection)")
    except ValueError:
        pass  # hostname is a domain name, not an IP — allowed

    return url
