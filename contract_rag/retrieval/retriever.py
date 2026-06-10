"""
retriever.py — Orchestrates hybrid search + reranking into a single call.
"""

import logging
from typing import Optional

import yaml
from qdrant_client import models

from contract_rag.retrieval.searcher import HybridSearcher
from contract_rag.retrieval.reranker import Reranker
from contract_rag.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, config: dict):
        self.config = config
        self.retrieval_cfg = config["retrieval"]

        self.embedder = Embedder(
            model_name=config["embedding"]["model_name"],
            batch_size=config["embedding"]["batch_size"],
        )
        self.searcher = HybridSearcher(config)
        self.reranker = Reranker(config)

        logger.info("Retriever initialised (embedder + searcher + reranker)")

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",          # "dense" | "sparse" | "hybrid"
        clause_type: Optional[str] = None,
        contract_type: Optional[str] = None,
        rerank: bool = True,
    ) -> list[dict]:
        """
        Full retrieval pipeline:
          1. Embed query (dense vector)
          2. Search (dense / sparse / hybrid)
          3. Rerank (cross-encoder)
          4. Return top-N enriched result dicts

        Each returned dict:
          {
            "id": str,
            "score": float,          # retrieval score
            "reranker_score": float, # cross-encoder score (if rerank=True)
            "payload": {
                "text": str,
                "clause_type": str,
                "contract_type": str,
                "parties": list,
                "source_file": str,
                "page_number": int,
                ...
            }
          }
        """
        if not query or not query.strip():
            logger.warning("retrieve called with empty query")
            return []

        logger.info("retrieve | mode=%s rerank=%s query='%s'", mode, rerank, query[:80])

        # 1. embed
        query_vector = self.embedder.embed_query(query)

        # 2. search
        if mode == "dense":
            results = self.searcher.dense_search(
                query_vector=query_vector,
                clause_type=clause_type,
                contract_type=contract_type,
            )
        elif mode == "sparse":
            results = self.searcher.sparse_search(
                query_text=query,
                clause_type=clause_type,
                contract_type=contract_type,
            )
        else:  # hybrid (default / production)
            results = self.searcher.hybrid_search(
                query_vector=query_vector,
                query_text=query,
                clause_type=clause_type,
                contract_type=contract_type,
            )

        if not results:
            logger.warning("No results returned from search (mode=%s)", mode)
            return []

        # 3. rerank
        if rerank:
            results = self.reranker.rerank(query=query, results=results)

        logger.info("retrieve complete — %d chunks returned", len(results))
        return results

    # ------------------------------------------------------------------
    # convenience: retrieve for a specific contract by cuad_id
    # ------------------------------------------------------------------

    def retrieve_for_contract(
        self,
        query: str,
        cuad_id: str,
        rerank: bool = True,
    ) -> list[dict]:
        if not query or not query.strip():
            return []

        query_vector = self.embedder.embed_query(query)

        contract_filter = models.Filter(
            must=[models.FieldCondition(key="source_file", match=models.MatchValue(value=cuad_id))]
        )
        sv = self.searcher._text_to_sparse(query)

        hits = self.searcher.client.query_points(
            collection_name=self.searcher.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=self.retrieval_cfg["dense_top_k"],
                    filter=contract_filter,
                ),
                models.Prefetch(
                    query=sv,
                    using="sparse",
                    limit=self.retrieval_cfg["sparse_top_k"],
                    filter=contract_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self.retrieval_cfg["dense_top_k"],
            with_payload=True,
        ).points

        results = [
            {"id": str(h.id), "score": h.score, "payload": h.payload or {}}
            for h in hits
        ]

        if rerank and results:
            results = self.reranker.rerank(query=query, results=results)

        logger.info(
            "retrieve_for_contract cuad_id=%s returned %d chunks", cuad_id, len(results)
        )
        return results


# ------------------------------------------------------------------
# module-level factory
# ------------------------------------------------------------------

def load_retriever(config_path: str = "contract_rag/configs/config.yaml") -> Retriever:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return Retriever(config)