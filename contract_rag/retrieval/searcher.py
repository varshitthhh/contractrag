"""
searcher.py — Hybrid search (dense + sparse BM25) over Qdrant with RRF fusion.
Compatible with qdrant-client >= 1.10
"""

import logging
import math
from typing import Optional

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


class HybridSearcher:
    def __init__(self, config: dict):
        qdrant_cfg = config["qdrant"]
        retrieval_cfg = config["retrieval"]

        self.client = QdrantClient(url=qdrant_cfg["url"])
        self.collection_name = qdrant_cfg["collection_name"]
        self.dense_top_k = retrieval_cfg["dense_top_k"]
        self.sparse_top_k = retrieval_cfg["sparse_top_k"]
        self.hybrid_alpha = retrieval_cfg["hybrid_alpha"]

        logger.info(
            "HybridSearcher ready — collection=%s dense_top_k=%d sparse_top_k=%d",
            self.collection_name,
            self.dense_top_k,
            self.sparse_top_k,
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        clause_type: Optional[str] = None,
        contract_type: Optional[str] = None,
    ) -> Optional[models.Filter]:
        conditions = []
        if clause_type:
            conditions.append(
                models.FieldCondition(
                    key="clause_type", match=models.MatchValue(value=clause_type)
                )
            )
        if contract_type:
            conditions.append(
                models.FieldCondition(
                    key="contract_type", match=models.MatchValue(value=contract_type)
                )
            )
        if not conditions:
            return None
        return models.Filter(must=conditions)

    def _text_to_sparse(self, text: str) -> models.SparseVector:
        tokens = text.lower().split()
        if not tokens:
            return models.SparseVector(indices=[], values=[])

        freq: dict[int, int] = {}
        for tok in tokens:
            idx = abs(hash(tok)) % (2**24)
            freq[idx] = freq.get(idx, 0) + 1

        total = len(tokens)
        indices = list(freq.keys())
        values = [math.log1p(f / total) for f in freq.values()]

        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        values = [v / norm for v in values]

        return models.SparseVector(indices=indices, values=values)

    def _hits_to_results(self, hits) -> list[dict]:
        return [
            {"id": str(h.id), "score": h.score, "payload": h.payload or {}}
            for h in hits
        ]

    # ------------------------------------------------------------------
    # public search methods
    # ------------------------------------------------------------------

    def dense_search(
        self,
        query_vector: list[float],
        top_k: Optional[int] = None,
        clause_type: Optional[str] = None,
        contract_type: Optional[str] = None,
    ) -> list[dict]:
        k = top_k or self.dense_top_k
        f = self._build_filter(clause_type, contract_type)

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            limit=k,
            query_filter=f,
            with_payload=True,
        ).points

        results = self._hits_to_results(hits)
        logger.debug("dense_search returned %d results", len(results))
        return results

    def sparse_search(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        clause_type: Optional[str] = None,
        contract_type: Optional[str] = None,
    ) -> list[dict]:
        k = top_k or self.sparse_top_k
        f = self._build_filter(clause_type, contract_type)
        sv = self._text_to_sparse(query_text)

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=sv,
            using="sparse",
            limit=k,
            query_filter=f,
            with_payload=True,
        ).points

        results = self._hits_to_results(hits)
        logger.debug("sparse_search returned %d results", len(results))
        return results

    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: Optional[int] = None,
        clause_type: Optional[str] = None,
        contract_type: Optional[str] = None,
    ) -> list[dict]:
        k = top_k or max(self.dense_top_k, self.sparse_top_k)
        f = self._build_filter(clause_type, contract_type)
        sv = self._text_to_sparse(query_text)

        hits = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=self.dense_top_k,
                    filter=f,
                ),
                models.Prefetch(
                    query=sv,
                    using="sparse",
                    limit=self.sparse_top_k,
                    filter=f,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        ).points

        results = self._hits_to_results(hits)
        logger.debug("hybrid_search returned %d results", len(results))
        return results

    def _manual_rrf(
        self,
        query_vector: list[float],
        query_text: str,
        k: int,
        query_filter: Optional[models.Filter],
    ) -> list[dict]:
        """Manual RRF fallback."""
        RRF_K = 60
        sv = self._text_to_sparse(query_text)

        dense_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            limit=self.dense_top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points

        sparse_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=sv,
            using="sparse",
            limit=self.sparse_top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points

        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for rank, h in enumerate(dense_hits):
            sid = str(h.id)
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (RRF_K + rank + 1)
            payloads[sid] = h.payload or {}

        for rank, h in enumerate(sparse_hits):
            sid = str(h.id)
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (RRF_K + rank + 1)
            payloads[sid] = h.payload or {}

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
        return [
            {"id": sid, "score": scores[sid], "payload": payloads[sid]}
            for sid in sorted_ids
        ]