from pathlib import Path
import pandas as pd
from pipelines.ingestion.extractors.pdf_extractor import ExtractedPage

SMALL_THRESHOLD = 1000
ROWS_PER_PAGE = 100


async def extract_csv(file_path: str) -> list[ExtractedPage]:
    path = Path(file_path)
    df = pd.read_csv(str(path))
    pages = []

    if len(df) <= SMALL_THRESHOLD:
        # Full markdown table
        md = df.to_markdown(index=False)
        pages.append(ExtractedPage(
            page_number=1,
            content=md,
            content_type="table",
            metadata={"source_file": path.name, "rows": len(df), "cols": len(df.columns)},
        ))
    else:
        # Statistical summary first
        summary_lines = [f"CSV file: {path.name}", f"Rows: {len(df)}, Columns: {len(df.columns)}", ""]
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].describe()
                summary_lines.append(
                    f"- {col} (numeric): min={s['min']:.2f}, max={s['max']:.2f}, mean={s['mean']:.2f}"
                )
            else:
                top = df[col].value_counts().head(5).to_dict()
                summary_lines.append(f"- {col} (categorical): top values = {top}")

        pages.append(ExtractedPage(
            page_number=1,
            content="\n".join(summary_lines),
            content_type="table",
            metadata={"source_file": path.name, "summary": True},
        ))

        # Chunked row blocks
        for i, start in enumerate(range(0, len(df), ROWS_PER_PAGE)):
            chunk = df.iloc[start:start + ROWS_PER_PAGE]
            pages.append(ExtractedPage(
                page_number=i + 2,
                content=chunk.to_markdown(index=False),
                content_type="table",
                metadata={"source_file": path.name, "row_start": start, "row_end": start + len(chunk)},
            ))

    return pages
