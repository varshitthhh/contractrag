"""
risk.py — Risk detection feature using retriever + generator.
"""

import logging
from contract_rag.retrieval.retriever import Retriever
from contract_rag.generation.generator import Generator, BaseResponse

logger = logging.getLogger(__name__)

RISK_QUERIES = [
    "liability cap limitation of liability",
    "indemnification indemnify hold harmless",
    "termination for cause convenience notice period",
    "auto-renewal automatic renewal evergreen clause",
    "confidentiality non-disclosure obligations",
]


class RiskAnalyser:
    def __init__(self, config: dict):
        self.retriever = Retriever(config)
        self.generator = Generator(config)
        logger.info("RiskAnalyser initialised")

    def analyse(
        self,
        cuad_id: str | None = None,
        contract_type: str | None = None,
    ) -> BaseResponse:
        """
        Retrieve risk-relevant clauses and run risk detection.

        If cuad_id is provided, scopes retrieval to that contract.
        Otherwise retrieves broadly filtered by contract_type.
        """
        all_chunks: list[dict] = []
        seen_ids: set[str] = set()

        for query in RISK_QUERIES:
            if cuad_id:
                chunks = self.retriever.retrieve_for_contract(
                    query=query,
                    cuad_id=cuad_id,
                    rerank=True,
                )
            else:
                chunks = self.retriever.retrieve(
                    query=query,
                    mode="hybrid",
                    contract_type=contract_type,
                    rerank=True,
                )

            for chunk in chunks:
                if chunk["id"] not in seen_ids:
                    seen_ids.add(chunk["id"])
                    all_chunks.append(chunk)

        # cap at 10 most relevant chunks to keep prompt manageable
        all_chunks = all_chunks[:10]

        logger.info(
            "RiskAnalyser retrieved %d unique chunks for cuad_id=%s",
            len(all_chunks),
            cuad_id,
        )

        return self.generator.risk(all_chunks)