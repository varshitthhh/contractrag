"""
models.py — Pydantic request/response schemas for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ------------------------------------------------------------------
# Shared
# ------------------------------------------------------------------

class CitationOut(BaseModel):
    chunk_index: int = 0
    clause: str = ""
    page: int = 0
    source_file: str = ""
    score: float = 0.0


class BaseContractResponse(BaseModel):
    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: float = 0.0
    risk_level: str = "NONE"
    flags: list[str] = Field(default_factory=list)
    supporting_chunks: list[str] = Field(default_factory=list)



class QualityScorecard(BaseModel):
    retrieval_score: float = 0.0
    grounding_score: float = 0.0
    faithfulness_flag: bool = False
    num_chunks_used: int = 0

# ------------------------------------------------------------------
# /api/v1/ask
# ------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Question to ask about the contract")
    clause_type: Optional[str] = Field(None, description="Filter by clause type")
    contract_type: Optional[str] = Field(None, description="Filter by contract type")
    mode: str = Field("hybrid", description="Retrieval mode: dense | sparse | hybrid")


class AskResponse(BaseContractResponse):
    quality_scorecard: Optional[QualityScorecard] = None


# ------------------------------------------------------------------
# /api/v1/risk
# ------------------------------------------------------------------

class RiskRequest(BaseModel):
    cuad_id: Optional[str] = Field(None, description="CUAD contract ID to scope analysis")
    contract_type: Optional[str] = Field(None, description="Filter by contract type")


class RiskResponse(BaseContractResponse):
    quality_scorecard: Optional[QualityScorecard] = None


# ------------------------------------------------------------------
# /api/v1/compare
# ------------------------------------------------------------------

class CompareRequest(BaseModel):
    vendor_cuad_id: str = Field(..., description="CUAD ID of vendor contract")
    template_cuad_id: str = Field(..., description="CUAD ID of template/baseline contract")


class ClauseDiffOut(BaseModel):
    clause_type: str = ""
    vendor_summary: str = ""
    template_summary: str = ""
    risk_delta: str = "EQUIVALENT"
    recommendation: str = ""


class CompareResponse(BaseModel):
    answer: str
    clause_diffs: list[ClauseDiffOut] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: float = 0.0
    risk_level: str = "NONE"
    flags: list[str] = Field(default_factory=list)
    quality_scorecard: Optional[QualityScorecard] = None


# ------------------------------------------------------------------
# /api/v1/health-score
# ------------------------------------------------------------------

class HealthRequest(BaseModel):
    cuad_id: str = Field(..., description="CUAD contract ID to score")


class ScoreBreakdownOut(BaseModel):
    completeness: int = 0
    risk_exposure: int = 0
    clarity: int = 0
    obligation_proximity: int = 0


class HealthResponse(BaseModel):
    answer: str
    health_score: int = Field(default=0, ge=0, le=100)
    score_breakdown: ScoreBreakdownOut = Field(default_factory=ScoreBreakdownOut)
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: float = 0.0
    risk_level: str = "NONE"
    flags: list[str] = Field(default_factory=list)
    quality_scorecard: Optional[QualityScorecard] = None


# ------------------------------------------------------------------
# /api/v1/evaluate
# ------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    config_path: str = Field(
        "contract_rag/configs/config.yaml",
        description="Path to config.yaml",
    )


class AblationRow(BaseModel):
    config: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    avg_latency_s: float


class EvaluateResponse(BaseModel):
    ablation_table: list[AblationRow]
    num_qa_pairs: int
    production_config: str




# ------------------------------------------------------------------
# Quality Scorecard (inline evaluation, no external calls)
# ------------------------------------------------------------------

class QualityScorecard(BaseModel):
    retrieval_score: float = Field(0.0, description='Avg reranker score of top chunks (0-1)')
    grounding_score: float = Field(0.0, description='Fraction of answer tokens found in context (0-1)')
    faithfulness_flag: bool = Field(False, description='True if answer stays within retrieved context')
    num_chunks_used: int = Field(0, description='Number of chunks used for generation')

# ------------------------------------------------------------------
# /health
# ------------------------------------------------------------------

class HealthCheckResponse(BaseModel):
    status: str
    qdrant: str
    ollama: str