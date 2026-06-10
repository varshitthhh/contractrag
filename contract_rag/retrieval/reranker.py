"""
reranker.py — Cross-encoder reranking using BAAI/bge-reranker-v2-m3.
"""

import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, config: dict):
        reranker_cfg = config["reranker"]
        self.model_name = reranker_cfg["model_name"]
        self.top_n = reranker_cfg["top_n"]

        logger.info("Loading reranker model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        logger.info("Reranker ready — top_n=%d", self.top_n)

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_n: int | None = None,
    ) -> list[dict]:
        n = top_n or self.top_n

        if not results:
            return []

        pairs = [
            [query, r["payload"].get("text", "")]
            for r in results
        ]

        try:
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
                if logits.shape[-1] > 1:
                    scores = torch.softmax(logits, dim=-1)[:, 1].tolist()
                else:
                    scores = torch.sigmoid(logits).squeeze(-1).tolist()

            if not isinstance(scores, list):
                scores = [scores]

        except Exception as e:
            logger.error("Reranker scoring failed: %s", e)
            for r in results:
                r["reranker_score"] = 0.0
            return results[:n]

        for r, score in zip(results, scores):
            r["reranker_score"] = float(score)

        reranked = sorted(results, key=lambda x: x["reranker_score"], reverse=True)
        top = reranked[:n]

        logger.debug(
            "rerank: %d → %d | top score=%.4f",
            len(results), len(top),
            top[0]["reranker_score"] if top else 0.0,
        )
        return top