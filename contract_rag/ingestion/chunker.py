import re
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CLAUSE_SEPARATORS = [
    r"\n(?=ARTICLE\s+[IVXLCDM]+)",
    r"\n(?=Section\s+\d+)",
    r"\n(?=\d+\.\s+[A-Z])",
    r"\n(?=[A-Z][A-Z\s]{4,}\.)",
    r"\n\n",
    r"\n",
    r" "
]

def chunk_contract(pages: list[dict], chunk_size: int = 512, chunk_overlap: int = 64) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CLAUSE_SEPARATORS,
        is_separator_regex=True
    )
    chunks = []
    chunk_index = 0
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            if len(split.strip()) < 50:
                continue
            chunks.append({
                "text": split.strip(),
                "page_number": page["page_number"],
                "source_file": page["source_file"],
                "chunk_index": chunk_index
            })
            chunk_index += 1
    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks