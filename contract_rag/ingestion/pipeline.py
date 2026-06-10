import logging
import yaml
from pathlib import Path
from qdrant_client import QdrantClient
from contract_rag.ingestion.parser import parse_contract
from contract_rag.ingestion.chunker import chunk_contract
from contract_rag.ingestion.metadata import extract_metadata
from contract_rag.ingestion.embedder import Embedder
from contract_rag.ingestion.indexer import init_collection, upsert_chunks

logger = logging.getLogger(__name__)

def load_config(config_path: str = "contract_rag/configs/config.yaml") -> dict:
    return yaml.safe_load(open(config_path))

def run_ingestion_pipeline(file_paths: list[str], config_path: str = "contract_rag/configs/config.yaml") -> dict:
    config = load_config(config_path)
    client = QdrantClient(url=config["qdrant"]["url"])
    embedder = Embedder(
        model_name=config["embedding"]["model_name"],
        batch_size=config["embedding"]["batch_size"]
    )
    init_collection(client, config["qdrant"]["collection_name"], config["qdrant"]["vector_size"])

    total_indexed = 0
    failed = []

    for file_path in file_paths:
        try:
            logger.info(f"Processing: {file_path}")
            pages = parse_contract(file_path)
            full_text = " ".join([p["text"] for p in pages])
            chunks = chunk_contract(pages, config["chunking"]["chunk_size"], config["chunking"]["chunk_overlap"])
            chunks = [extract_metadata(c, full_text) for c in chunks]
            texts = [c["text"] for c in chunks]
            embeddings = embedder.embed(texts)
            source_file = Path(file_path).name
            n = upsert_chunks(client, config["qdrant"]["collection_name"], chunks, embeddings, source_file)
            total_indexed += n
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            failed.append(file_path)

    return {"total_indexed": total_indexed, "failed": failed, "processed": len(file_paths) - len(failed)}

if __name__ == "__main__":
    import glob
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    sample_size = config["data"]["cuad_sample_size"]
    files = glob.glob(f"{raw_dir}/**/*.pdf", recursive=True)[:sample_size]
    if not files:
        print(f"No PDFs found in {raw_dir}")
    else:
        result = run_ingestion_pipeline(files)
        print(f"Ingestion complete: {result}")
