from dataclasses import dataclass, field
from pathlib import Path
import pypdfium2 as pdfium
import pdfplumber
import pytesseract
from PIL import Image
import io


@dataclass
class ExtractedPage:
    page_number: int
    content: str
    content_type: str  # "text" | "table" | "image_description"
    metadata: dict = field(default_factory=dict)


async def extract_pdf(file_path: str) -> list[ExtractedPage]:
    pages: list[ExtractedPage] = []
    path = Path(file_path)

    pdf = pdfium.PdfDocument(str(path))

    with pdfplumber.open(str(path)) as plumber_pdf:
        for page_num in range(len(pdf)):
            pdfium_page = pdf[page_num]
            plumber_page = plumber_pdf.pages[page_num]

            # Extract text via pdfium
            text_page = pdfium_page.get_textpage()
            text = text_page.get_text_range()
            text_page.close()

            # Determine if scanned (low text density)
            is_scanned = len(text.strip()) < 50

            if is_scanned:
                # Rasterize and OCR
                bitmap = pdfium_page.render(scale=2)
                pil_image = bitmap.to_pil()
                text = pytesseract.image_to_string(pil_image, config="--psm 6")

            if text.strip():
                pages.append(ExtractedPage(
                    page_number=page_num + 1,
                    content=text.strip(),
                    content_type="text",
                    metadata={"scanned": is_scanned, "source_file": path.name},
                ))

            # Extract tables via pdfplumber
            tables = plumber_page.extract_tables()
            for table_idx, table in enumerate(tables):
                if not table:
                    continue
                md_rows = []
                for row_idx, row in enumerate(table):
                    cells = [str(c or "") for c in row]
                    md_rows.append("| " + " | ".join(cells) + " |")
                    if row_idx == 0:
                        md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
                md_table = "\n".join(md_rows)
                pages.append(ExtractedPage(
                    page_number=page_num + 1,
                    content=md_table,
                    content_type="table",
                    metadata={"table_index": table_idx, "source_file": path.name},
                ))

    pdf.close()
    return pages
