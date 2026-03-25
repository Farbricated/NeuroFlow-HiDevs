import re
import logging

logger = logging.getLogger(__name__)

SECRET_PATTERNS = [
    ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("api_key", re.compile(r'["\']?(?:api|secret|token|key|password)["\']?\s*[:=]\s*["\'][A-Za-z0-9/+]{20,}["\']', re.IGNORECASE)),
    ("pem_header", re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")),
]

REDACT_PLACEHOLDER = "[REDACTED]"


def scan_and_redact(text: str, document_id: str = "") -> tuple[str, list[dict]]:
    """Scan text for secrets and redact them. Returns (redacted_text, findings)."""
    findings = []
    result = text
    for name, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(result):
            findings.append({
                "event": "secret_redacted",
                "document_id": document_id,
                "pattern_type": name,
            })
            logger.warning(
                "Secret detected and redacted: type=%s document_id=%s", name, document_id
            )
        result = pattern.sub(REDACT_PLACEHOLDER, result)
    return result, findings
