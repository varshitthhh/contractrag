"""
compare.py — Contract comparison feature using retriever + generator.
"""

import logging
from contract_rag.retrieval.retriever import Retriever
from contract_rag.generation.generator import Generator, CompareResponse

logger = logging.getLogger(__name__)

COMPARE_QUERIES = [
    "liability cap limitation of liability damages",
    "indemnification indemnify hold harmless obligations",
    "termination for cause convenience notice period",
    "confidentiality non-disclosure obligations",
    "governing law jurisdiction dispute resolution",
    "payment terms invoicing fees",
    "intellectual property ownership assignment",
    "warranties representations guarantees",
]


class ContractComparer:
    def __init__(self, config: dict):
        self.retriever = Retriever(config)
        self.generator = Generator(config)
        logger.info("ContractComparer initialised")

    def compare(
        self,
        vendor_cuad_id: str,
        template_cuad_id: str,
    ) -> CompareResponse:
        """
        Retrieve clauses from both contracts and compare them clause by clause.

        vendor_cuad_id   — the contract being reviewed (e.g. vendor-supplied)
        template_cuad_id — the baseline/template contract (e.g. your standard form)
        """
        vendor_chunks: list[dict] = []
        template_chunks: list[dict] = []
        seen_vendor: set[str] = set()
        seen_template: set[str] = set()

        for query in COMPARE_QUERIES:
            # vendor side
            v_chunks = self.retriever.retrieve_for_contract(
                query=query,
                cuad_id=vendor_cuad_id,
                rerank=True,
            )
            for chunk in v_chunks:
                if chunk["id"] not in seen_vendor:
                    seen_vendor.add(chunk["id"])
                    vendor_chunks.append(chunk)

            # template side
            t_chunks = self.retriever.retrieve_for_contract(
                query=query,
                cuad_id=template_cuad_id,
                rerank=True,
            )
            for chunk in t_chunks:
                if chunk["id"] not in seen_template:
                    seen_template.add(chunk["id"])
                    template_chunks.append(chunk)

        # cap each side at 8 chunks to keep prompt size reasonable
        vendor_chunks = vendor_chunks[:8]
        template_chunks = template_chunks[:8]

        logger.info(
            "ContractComparer: vendor=%d chunks template=%d chunks",
            len(vendor_chunks),
            len(template_chunks),
        )

        return self.generator.compare(vendor_chunks, template_chunks)