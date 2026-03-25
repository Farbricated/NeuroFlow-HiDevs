import base64
from pathlib import Path
from PIL import Image
import pytesseract
from pipelines.ingestion.extractors.pdf_extractor import ExtractedPage
from backend.providers.base import ChatMessage, RoutingCriteria
from backend.providers.router import ModelRouter

MAX_DIM = 1024


async def extract_image(file_path: str) -> list[ExtractedPage]:
    path = Path(file_path)
    img = Image.open(str(path))

    # Resize to max dimension
    w, h = img.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # OCR for embedded text
    ocr_text = pytesseract.image_to_string(img).strip()

    # Vision LLM description
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    router = ModelRouter()
    provider = await router.select(RoutingCriteria(require_vision=True))

    messages = [ChatMessage(
        role="user",
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            },
            {
                "type": "text",
                "text": (
                    "Describe this image in detail. Include all visible text, diagrams, "
                    "charts, objects, and their relationships. Be thorough and precise."
                ),
            },
        ],
    )]

    result = await provider.complete(messages, max_tokens=1024)
    description = result.content

    combined = description
    if ocr_text:
        combined += f"\n\nText found in image:\n{ocr_text}"

    return [ExtractedPage(
        page_number=1,
        content=combined,
        content_type="image_description",
        metadata={"source_file": path.name, "ocr_text": ocr_text},
    )]
