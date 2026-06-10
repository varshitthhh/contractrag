"""
health.py — Contract health scoring feature using retriever + generator.
"""

import logging
from contract_rag.retrieval.retriever import Retriever
from contract_rag.generation.generator import Generator, HealthResponse

logger = logging.getLogger(__name__)

HEALTH_QUERIES = [
    "liability cap limitation of liability damages",
    "indemnification indemnify hold harmless",
    "termination for cause convenience notice period",
    "confidentiality non-disclosure obligations",
    "governing law jurisdiction dispute resolution",
    "payment terms invoicing fees penalties",
    "intellectual property ownership assignment license",
    "warranties representations guarantees",
    "renewal expiration effective date term",
    "obligations duties responsibilities parties",
]


class HealthScorer:
    def __init__(self, config: dict):
        self.retriever = Retriever(config)
        self.generator = Generator(config)
        logger.info("HealthScorer initialised")

    def score(self, cuad_id: str) -> HealthResponse:
        """
        Retrieve all major clause types from a contract and
        generate a 0-100 health score with breakdown.
        """
        all_chunks: list[dict] = []
        seen_ids: set[str] = set()

        for query in HEALTH_QUERIES:
            chunks = self.retriever.retrieve_for_contract(
                query=query,
                cuad_id=cuad_id,
                rerank=True,
            )
            for chunk in chunks:
                if chunk["id"] not in seen_ids:
                    seen_ids.add(chunk["id"])
                    all_chunks.append(chunk)

        # cap at 10 chunks — enough coverage, keeps prompt tight
        all_chunks = all_chunks[:10]

        logger.info(
            "HealthScorer retrieved %d unique chunks for cuad_id=%s",
            len(all_chunks),
            cuad_id,
        )

        return self.generator.health(all_chunks)