"""
ragas_eval.py — RAGAS evaluation pipeline with 4-config ablation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from contract_rag.retrieval.retriever import Retriever
from contract_rag.generation.generator import Generator

logger = logging.getLogger(__name__)

ABLATION_CONFIGS = [
    {"name": "Dense",           "mode": "dense",  "rerank": False},
    {"name": "BM25",            "mode": "sparse", "rerank": False},
    {"name": "Hybrid",          "mode": "hybrid", "rerank": False},
    {"name": "Hybrid+Reranker", "mode": "hybrid", "rerank": True},
]


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _load_gold_qa(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def _run_single_config(
    config: dict,
    gold_qa: list[dict],
    retriever: Retriever,
    generator: Generator,
    ablation_cfg: dict,
) -> dict[str, Any]:
    """
    Run retrieval + generation for all gold QA pairs under one ablation config.
    Returns a dict ready for RAGAS Dataset construction.
    """
    questions, answers, contexts, ground_truths = [], [], [], []
    latencies = []

    for qa in gold_qa:
        question = qa["question"]
        ground_truth = qa["ground_truth"]
        clause_type = qa.get("clause_type")

        t0 = time.time()
        chunks = retriever.retrieve(
            query=question,
            mode=ablation_cfg["mode"],
            clause_type=clause_type,
            rerank=ablation_cfg["rerank"],
        )
        response = generator.ask(question=question, chunks=chunks)
        latency = time.time() - t0

        questions.append(question)
        answers.append(response.answer)
        contexts.append([c["payload"].get("text", "") for c in chunks])
        ground_truths.append(ground_truth)
        latencies.append(latency)

        logger.debug(
            "QA %s | mode=%s rerank=%s | latency=%.2fs",
            qa["id"],
            ablation_cfg["mode"],
            ablation_cfg["rerank"],
            latency,
        )

    return {
        "questions": questions,
        "answers": answers,
        "contexts": contexts,
        "ground_truths": ground_truths,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def _ragas_score(data: dict) -> dict[str, float]:
    """Run RAGAS metrics on a completed data dict."""
    dataset = Dataset.from_dict({
        "question":   data["questions"],
        "answer":     data["answers"],
        "contexts":   data["contexts"],
        "ground_truth": data["ground_truths"],
    })

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return {
        "faithfulness":      round(float(result["faithfulness"]), 4),
        "answer_relevancy":  round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
        "context_recall":    round(float(result["context_recall"]), 4),
    }


# ------------------------------------------------------------------
# main evaluation runner
# ------------------------------------------------------------------

def run_evaluation(config_path: str = "contract_rag/configs/config.yaml") -> dict:
    """
    Run full ablation evaluation across 4 retrieval configs.
    Returns ablation table as a dict and saves to processed dir.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    gold_qa_path = Path("contract_rag/evaluation/gold_qa.json")
    gold_qa = _load_gold_qa(str(gold_qa_path))
    logger.info("Loaded %d gold QA pairs", len(gold_qa))

    retriever = Retriever(config)
    generator = Generator(config)

    ablation_results = []

    for ablation_cfg in ABLATION_CONFIGS:
        logger.info("Running ablation config: %s", ablation_cfg["name"])

        data = _run_single_config(
            config=config,
            gold_qa=gold_qa,
            retriever=retriever,
            generator=generator,
            ablation_cfg=ablation_cfg,
        )

        scores = _ragas_score(data)
        scores["avg_latency_s"] = round(data["avg_latency"], 3)
        scores["config"] = ablation_cfg["name"]

        ablation_results.append(scores)
        logger.info("Config=%s scores=%s", ablation_cfg["name"], scores)

    # build output
    output = {
        "ablation_table": ablation_results,
        "num_qa_pairs": len(gold_qa),
        "production_config": "Hybrid+Reranker",
    }

    # save results
    out_path = Path(config["data"]["processed_dir"]) / "ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Ablation results saved to %s", out_path)
    _print_table(ablation_results)

    return output


def _print_table(results: list[dict]) -> None:
    """Pretty-print ablation table to stdout."""
    header = f"{'Config':<20} {'Faithfulness':>13} {'Ans.Relevancy':>14} {'Ctx.Precision':>14} {'Ctx.Recall':>11} {'Latency(s)':>11}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(
            f"{r['config']:<20} "
            f"{r['faithfulness']:>13.4f} "
            f"{r['answer_relevancy']:>14.4f} "
            f"{r['context_precision']:>14.4f} "
            f"{r['context_recall']:>11.4f} "
            f"{r['avg_latency_s']:>11.3f}"
        )
    print("=" * len(header) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()