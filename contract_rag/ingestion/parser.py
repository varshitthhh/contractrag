import logging
from pathlib import Path
from typing import Optional
import pymupdf
from docx import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def parse_pdf(file_path: str) -> list[dict]:
    pages = []
    try:
        doc = pymupdf.open(file_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({"page_number": page_num, "text": text, "source_file": Path(file_path).name})
        doc.close()
        logger.info(f"Parsed {len(pages)} pages from {file_path}")
    except Exception as e:
        logger.error(f"Failed to parse PDF {file_path}: {e}")
        raise
    return pages

def parse_docx(file_path: str) -> list[dict]:
    pages = []
    try:
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        pages.append({"page_number": 1, "text": text, "source_file": Path(file_path).name})
        logger.info(f"Parsed DOCX {file_path}")
    except Exception as e:
        logger.error(f"Failed to parse DOCX {file_path}: {e}")
        raise
    return pages

def parse_contract(file_path: str) -> list[dict]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
