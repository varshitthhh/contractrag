"""
routes.py — FastAPI route handlers for all ContractRAG endpoints.
"""

import logging
from fastapi import APIRouter, HTTPException

from contract_rag.api.models import (
    QualityScorecard,
    AskRequest, AskResponse,
    RiskRequest, RiskResponse,
    CompareRequest, CompareResponse,
    HealthRequest, HealthResponse,
    EvaluateRequest, EvaluateResponse,
    HealthCheckResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# dependency: app state injected via request.app.state
# ------------------------------------------------------------------

def get_retriever(request):
    return request.app.state.retriever

def get_generator(request):
    return request.app.state.generator

def get_risk_analyser(request):
    return request.app.state.risk_analyser

def get_comparer(request):
    return request.app.state.comparer

def get_health_scorer(request):
    return request.app.state.health_scorer

def get_llm(request):
    return request.app.state.llm


# ------------------------------------------------------------------
# /health
# ------------------------------------------------------------------

from fastapi import Request

@router.get("/health", response_model=HealthCheckResponse, tags=["system"])
def health_check(request: Request):
    llm = get_llm(request)
    retriever = get_retriever(request)

    ollama_ok = llm.health_check()

    try:
        retriever.searcher.client.get_collection(
            retriever.searcher.collection_name
        )
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    return HealthCheckResponse(
        status="ok" if (ollama_ok and qdrant_ok) else "degraded",
        qdrant="ok" if qdrant_ok else "unreachable",
        ollama="ok" if ollama_ok else "unreachable",
    )


# ------------------------------------------------------------------
# /api/v1/ask
# ------------------------------------------------------------------

@router.post("/api/v1/ask", response_model=AskResponse, tags=["qa"])
def ask(request: Request, body: AskRequest):
    retriever = get_retriever(request)
    generator = get_generator(request)

    try:
        chunks = retriever.retrieve(
            query=body.question,
            mode=body.mode,
            clause_type=body.clause_type,
            contract_type=body.contract_type,
            rerank=True,
        )
        result = generator.ask(question=body.question, chunks=chunks)
    except Exception as e:
        logger.error("/ask error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    from contract_rag.evaluation.scorer import compute_scorecard
    citation_scores = [{"score": c.get("reranker_score", c.get("score", 0.0))} for c in chunks]
    sc = compute_scorecard(result.answer, chunks, citation_scores)
    dump = result.model_dump()
    dump['quality_scorecard'] = sc
    return AskResponse(**dump)


# ------------------------------------------------------------------
# /api/v1/risk
# ------------------------------------------------------------------

@router.post("/api/v1/risk", response_model=RiskResponse, tags=["risk"])
def risk(request: Request, body: RiskRequest):
    risk_analyser = get_risk_analyser(request)

    try:
        result = risk_analyser.analyse(
            cuad_id=body.cuad_id,
            contract_type=body.contract_type,
        )
    except Exception as e:
        logger.error("/risk error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    from contract_rag.evaluation.scorer import compute_scorecard
    sc = compute_scorecard(result.answer, [{'text': c.clause} for c in result.citations], result.citations)
    return RiskResponse(**result.model_dump(), quality_scorecard=QualityScorecard(**sc))


# ------------------------------------------------------------------
# /api/v1/compare
# ------------------------------------------------------------------

@router.post("/api/v1/compare", response_model=CompareResponse, tags=["compare"])
def compare(request: Request, body: CompareRequest):
    comparer = get_comparer(request)

    try:
        result = comparer.compare(
            vendor_cuad_id=body.vendor_cuad_id,
            template_cuad_id=body.template_cuad_id,
        )
    except Exception as e:
        logger.error("/compare error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    from contract_rag.evaluation.scorer import compute_scorecard
    sc = compute_scorecard(result.answer, [{'text': c.clause} for c in result.citations], result.citations)
    return CompareResponse(**result.model_dump(), quality_scorecard=QualityScorecard(**sc))


# ------------------------------------------------------------------
# /api/v1/health-score
# ------------------------------------------------------------------

@router.post("/api/v1/health-score", response_model=HealthResponse, tags=["health"])
def health_score(request: Request, body: HealthRequest):
    health_scorer = get_health_scorer(request)

    try:
        result = health_scorer.score(cuad_id=body.cuad_id)
    except Exception as e:
        logger.error("/health-score error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    from contract_rag.evaluation.scorer import compute_scorecard
    sc = compute_scorecard(result.answer, [{'text': c.clause} for c in result.citations], result.citations)
    return HealthResponse(**result.model_dump(), quality_scorecard=QualityScorecard(**sc))


# ------------------------------------------------------------------
# /api/v1/evaluate  (internal)
# ------------------------------------------------------------------

@router.post("/api/v1/evaluate", response_model=EvaluateResponse, tags=["eval"])
def evaluate(request: Request, body: EvaluateRequest):
    from contract_rag.evaluation.ragas_eval import run_evaluation

    try:
        result = run_evaluation(config_path=body.config_path)
    except Exception as e:
        logger.error("/evaluate error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return EvaluateResponse(**result)