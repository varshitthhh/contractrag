"""
generator.py — Prompt builder + LLM call + Pydantic output parser.
"""

import json
import logging
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from contract_rag.generation.llm import OllamaClient
from contract_rag.generation.prompts import (
    build_ask_prompt,
    build_risk_prompt,
    build_compare_prompt,
    build_health_prompt,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Pydantic output schemas
# ------------------------------------------------------------------

class CitationModel(BaseModel):
    chunk_index: int = 0
    clause: str = ""
    page: int = 0
    source_file: str = ""
    score: float = 0.0


class BaseResponse(BaseModel):
    answer: str
    citations: list[CitationModel] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: str = Field(default="NONE")
    flags: list[str] = Field(default_factory=list)
    supporting_chunks: list[str] = Field(default_factory=list)

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, v: str) -> str:
        allowed = {"HIGH", "MEDIUM", "LOW", "NONE"}
        return v.upper() if v.upper() in allowed else "NONE"

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class ClauseDiff(BaseModel):
    clause_type: str = ""
    vendor_summary: str = ""
    template_summary: str = ""
    risk_delta: str = "EQUIVALENT"
    recommendation: str = ""


class CompareResponse(BaseModel):
    answer: str
    clause_diffs: list[ClauseDiff] = Field(default_factory=list)
    citations: list[CitationModel] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: str = Field(default="NONE")
    flags: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    completeness: int = 0
    risk_exposure: int = 0
    clarity: int = 0
    obligation_proximity: int = 0


class HealthResponse(BaseModel):
    answer: str
    health_score: int = Field(default=0, ge=0, le=100)
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    citations: list[CitationModel] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: str = Field(default="NONE")
    flags: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------
# JSON extraction helper
# ------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Robustly extract the first JSON object from LLM output.
    Handles markdown fences, leading text, trailing junk, double-encoded JSON.
    """
    # strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # find first { ... } block
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")

    depth = 0
    end = start
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    raw = text[start: end + 1] if depth == 0 else text[start:]
    if depth > 0:
        open_brackets = raw.count('[') - raw.count(']')
        open_braces = raw.count('{') - raw.count('}')
        if open_brackets > 0:
            raw += ']' * open_brackets
        if open_braces > 0:
            raw += '}' * open_braces
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}\nRaw: {raw[:300]}")

    # handle double-encoded JSON — if "answer" is itself a JSON string
    if isinstance(parsed.get("answer"), str) and parsed["answer"].strip().startswith("{"):
        try:
            inner = json.loads(parsed["answer"])
            if isinstance(inner, dict):
                return inner
        except Exception:
            pass

    return parsed


# ------------------------------------------------------------------
# Generator
# ------------------------------------------------------------------

class Generator:
    def __init__(self, config: dict):
        self.llm = OllamaClient(config)
        logger.info("Generator initialised")

    def _call_llm(self, prompt: str) -> dict:
        raw = self.llm.generate(prompt)
        try:
            return _extract_json(raw)
        except ValueError as e:
            logger.error("JSON extraction failed: %s", e)
            return {
                "answer": raw[:500] if raw else "LLM returned no output.",
                "confidence": 0.0,
                "risk_level": "NONE",
                "flags": ["JSON parse failed — raw LLM output returned"],
                "citations": [],
                "supporting_chunks": [],
            }

    # ------------------------------------------------------------------
    # public task methods
    # ------------------------------------------------------------------

    def ask(self, question: str, chunks: list[dict]) -> BaseResponse:
        """Free-form Q&A over retrieved contract chunks."""
        if not chunks:
            return BaseResponse(
                answer="No relevant contract clauses found for this question.",
                flags=["No context retrieved"],
            )
        prompt = build_ask_prompt(question, chunks)
        data = self._call_llm(prompt)
        try:
            return BaseResponse(**data)
        except Exception as e:
            logger.error("BaseResponse validation error: %s", e)
            return BaseResponse(answer=data.get("answer", "Parse error"), flags=[str(e)])

    def risk(self, chunks: list[dict]) -> BaseResponse:
        """Risk detection over contract chunks."""
        if not chunks:
            return BaseResponse(
                answer="No contract clauses found for risk analysis.",
                flags=["No context retrieved"],
                risk_level="NONE",
            )
        prompt = build_risk_prompt(chunks)
        data = self._call_llm(prompt)
        try:
            return BaseResponse(**data)
        except Exception as e:
            logger.error("Risk response validation error: %s", e)
            return BaseResponse(answer=data.get("answer", "Parse error"), flags=[str(e)])

    def compare(
        self,
        vendor_chunks: list[dict],
        template_chunks: list[dict],
    ) -> CompareResponse:
        """Clause-level comparison between vendor and template contract."""
        if not vendor_chunks or not template_chunks:
            return CompareResponse(
                answer="Insufficient context for comparison.",
                flags=["Missing vendor or template chunks"],
            )
        prompt = build_compare_prompt(vendor_chunks, template_chunks)
        data = self._call_llm(prompt)
        try:
            return CompareResponse(**data)
        except Exception as e:
            logger.error("CompareResponse validation error: %s", e)
            return CompareResponse(
                answer=data.get("answer", "Parse error"), flags=[str(e)]
            )

    def health(self, chunks: list[dict]) -> HealthResponse:
        """Health score generation for a contract."""
        if not chunks:
            return HealthResponse(
                answer="No contract clauses found for health scoring.",
                health_score=0,
                flags=["No context retrieved"],
            )
        prompt = build_health_prompt(chunks)
        data = self._call_llm(prompt)
        try:
            return HealthResponse(**data)
        except Exception as e:
            logger.error("HealthResponse validation error: %s", e)
            return HealthResponse(
                answer=data.get("answer", "Parse error"),
                health_score=0,
                flags=[str(e)],
            )