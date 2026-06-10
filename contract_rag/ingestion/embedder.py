import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", batch_size: int = 32):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        logger.info("Embedding model loaded")

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def embed_query(self, query: str) -> list[float]:
        try:
            embedding = self.model.encode(
                f"Represent this sentence for searching relevant passages: {query}",
                normalize_embeddings=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise